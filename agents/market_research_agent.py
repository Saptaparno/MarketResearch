import json
import os
import sys
from pathlib import Path
from typing import Any
import asyncio
import re
import pandas as pd
import yfinance as yf
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools import tool
from agno.db.postgres import AsyncPostgresDb
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector
from agno.run import RunContext

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from models.schema import MarketResearchAgentOutput, MarketResearchContent
load_dotenv()

KNOWLEDGE_BASE_DIR = Path(os.getenv("KNOWLEDGE_BASE_DIR", _PROJECT_ROOT / "knowledge_base")).resolve()


def _postgres_url_for_async(db_url: str | None) -> str | None:
    if db_url is None:
        return None
    # AsyncPostgresDb needs an async driver; plain postgresql:// loads psycopg2 (sync).
    if "+" in db_url.split("://", 1)[0]:
        return db_url
    if db_url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + db_url.removeprefix("postgresql://")
    if db_url.startswith("postgres://"):
        return "postgresql+asyncpg://" + db_url.removeprefix("postgres://")
    return db_url


def _postgres_url_for_pgvector(db_url: str | None) -> str | None:
    """PgVector uses sync SQLAlchemy in Knowledge(); use postgresql+psycopg (not +asyncpg)."""
    if db_url is None:
        return None
    if db_url.startswith("postgresql+asyncpg://"):
        db_url = "postgresql://" + db_url.removeprefix("postgresql+asyncpg://")
    if db_url.startswith("postgres+asyncpg://"):
        db_url = "postgres://" + db_url.removeprefix("postgres+asyncpg://")
    scheme = db_url.split("://", 1)[0]
    if "+" in scheme:
        return db_url
    if db_url.startswith("postgresql://"):
        return "postgresql+psycopg://" + db_url.removeprefix("postgresql://")
    if db_url.startswith("postgres://"):
        return "postgresql+psycopg://" + db_url.removeprefix("postgres://")
    return db_url


db_client = _postgres_url_for_async(os.getenv("DATABASE_URL"))
knowledgebase_pg_url = _postgres_url_for_pgvector(os.getenv("KNOWLEDGEBASE_DATABASE_URL"))
db = AsyncPostgresDb(id="market-research-db", db_url=db_client)
vectordb = PgVector(table_name="ai_knowledge_base", db_url=knowledgebase_pg_url)
knowledge_base = Knowledge(vector_db=vectordb)


def canonical_kb_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


async def maybe_embed_runtime_kb_payload(
    *,
    kb_name: str,
    payload: dict[str, Any],
    metadata: dict[str, Any],
    description: str | None = None,
) -> None:
    """Store runtime tool payloads in pgvector when this exact content hash is not already indexed."""
    if knowledgebase_pg_url is None:
        return
    text = canonical_kb_json(payload)
    if not text.strip():
        return
    try:
        await knowledge_base.ainsert(
            name=kb_name,
            description=description,
            text_content=text,
            metadata=metadata,
            skip_if_exists=True,
        )
    except Exception:
        pass


def _yahoo_symbol(raw: str) -> str:
    s = raw.strip()
    m = re.search(r"\(([A-Za-z0-9^.-]{1,20})\)", s)
    return (m.group(1) if m else s.split()[0]).upper()


def frame_to_records_dict(df):
    tbl = df.reset_index()
    if isinstance(tbl.columns, pd.MultiIndex):
        tbl.columns = [
            "_".join(str(part) for part in col).strip("_")
            if isinstance(col, tuple)
            else str(col)
            for col in tbl.columns
        ]
    return {"records": json.loads(tbl.to_json(orient="records", date_format="iso"))}


@tool
async def get_historical_data(
    run_context: RunContext, ticker: str, period: str = "1y", interval: str = "1d"
):
    ticker = _yahoo_symbol(ticker)
    data = await asyncio.to_thread(yf.download, ticker, period=period, interval=interval)
    payload = frame_to_records_dict(data)
    run_context.session_state["historical_data"] = payload
    if payload.get("records"):
        await maybe_embed_runtime_kb_payload(
            kb_name=f"runtime:yfinance:{ticker}:historical:{period}:{interval}",
            description=f"Yahoo Finance OHLCV {ticker} ({period}, {interval})",
            payload=payload,
            metadata={
                "source": "yfinance",
                "kind": "historical",
                "ticker": ticker,
                "period": period,
                "interval": interval,
            },
        )
    return MarketResearchContent(historical_data=payload)


@tool
async def get_stock_info(run_context: RunContext, ticker: str):
    ticker = _yahoo_symbol(ticker)
    stock = yf.Ticker(ticker)
    info = await asyncio.to_thread(getattr, stock, "info")
    run_context.session_state["stock_info"] = info
    if isinstance(info, dict) and info:
        await maybe_embed_runtime_kb_payload(
            kb_name=f"runtime:yfinance:{ticker}:stock_info",
            description=f"Yahoo Finance fundamentals/info snapshot for {ticker}",
            payload={"stock_info": info},
            metadata={"source": "yfinance", "kind": "stock_info", "ticker": ticker},
        )
    return MarketResearchContent(stock_info=info)


@tool
async def get_analyst_recommendations(run_context: RunContext, ticker: str):
    ticker = _yahoo_symbol(ticker)
    stock = yf.Ticker(ticker)
    recommendations = await asyncio.to_thread(getattr, stock, "recommendations")
    if recommendations is None or recommendations.empty:
        run_context.session_state["analyst_recommendations"] = {}
        return MarketResearchContent(analyst_recommendations={})
    payload = frame_to_records_dict(recommendations)
    run_context.session_state["analyst_recommendations"] = payload
    if payload.get("records"):
        await maybe_embed_runtime_kb_payload(
            kb_name=f"runtime:yfinance:{ticker}:analyst_recommendations",
            description=f"Yahoo Finance analyst recommendations history for {ticker}",
            payload=payload,
            metadata={"source": "yfinance", "kind": "analyst_recommendations", "ticker": ticker},
        )
    return MarketResearchContent(analyst_recommendations=payload)


async def merge_market_output_from_session_state(run_output, run_context, **_kwargs):
    state = run_context.session_state or {}
    run_output.content = MarketResearchAgentOutput(
        market_research=MarketResearchContent(
            historical_data=state.get("historical_data"),
            stock_info=state.get("stock_info"),
            analyst_recommendations=state.get("analyst_recommendations"),
        )
    )


def market_research_tools_factory(agent=None, run_context=None):
    return [get_historical_data, get_stock_info, get_analyst_recommendations]


def create_market_research_agent(team=None, run_context=None):
    return Agent(
        db=db,
        knowledge=knowledge_base,
        search_knowledge=True,
        add_knowledge_to_context=True,
        name="market_research_agent",
        description="A market research agent that can help with market research",
        reasoning=True,
        tools=market_research_tools_factory,
        add_history_to_context=True,
        add_session_state_to_context=True,
        session_state={
            "historical_data": None,
            "stock_info": None,
            "analyst_recommendations": None,
        },

        output_schema=MarketResearchAgentOutput,
        post_hooks=[merge_market_output_from_session_state],
        model=OpenAIResponses(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
    )


market_research_agent = create_market_research_agent()