import json
import os
import sys
from pathlib import Path
import asyncio
import re
import pandas as pd
import yfinance as yf
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools import tool
from agno.db.postgres import AsyncPostgresDb
from agno.run import RunContext

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.schema import MarketResearchData

load_dotenv()

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


db_client = _postgres_url_for_async(os.getenv("DATABASE_URL"))
db = AsyncPostgresDb(db_url=db_client)

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
    return MarketResearchData(historical_data=payload)


@tool
async def get_stock_info(run_context: RunContext, ticker: str):
    ticker = _yahoo_symbol(ticker)
    stock = yf.Ticker(ticker)
    info = await asyncio.to_thread(getattr, stock, "info")
    run_context.session_state["stock_info"] = info
    return MarketResearchData(stock_info=info)


@tool
async def get_analyst_recommendations(run_context: RunContext, ticker: str):
    ticker = _yahoo_symbol(ticker)
    stock = yf.Ticker(ticker)
    recommendations = await asyncio.to_thread(getattr, stock, "recommendations")
    if recommendations is None or recommendations.empty:
        run_context.session_state["analyst_recommendations"] = {}
        return MarketResearchData(analyst_recommendations={})
    payload = frame_to_records_dict(recommendations)
    run_context.session_state["analyst_recommendations"] = payload
    return MarketResearchData(analyst_recommendations=payload)


async def merge_market_output_from_session_state(run_output, run_context, **_kwargs):
    """Native structured-output content ignores tool payloads; overwrite from session_state."""
    state = run_context.session_state or {}
    run_output.content = MarketResearchData(
        historical_data=state.get("historical_data"),
        stock_info=state.get("stock_info"),
        analyst_recommendations=state.get("analyst_recommendations"),
    )


market_research_agent = Agent(
    db=db,
    name="market_research_agent", 
    description="A market research agent that can help with market research", 
    tools=[get_historical_data, get_stock_info, get_analyst_recommendations],
    add_history_to_context=True, 
    add_session_state_to_context=True,
    session_state={
        "historical_data": None,
        "stock_info": None,
        "analyst_recommendations": None,
    },
    output_schema=MarketResearchData,
    post_hooks=[merge_market_output_from_session_state],
    model=OpenAIResponses(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")))