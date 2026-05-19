import asyncio
import os
import re
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools import tool
from agno.run import RunContext
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from agents.market_research_agent import db, frame_to_records_dict, knowledge_base, maybe_embed_runtime_kb_payload
from models.schema import PortfolioHealthAgentOutput, PortfolioHealthContent

load_dotenv()


def _symbol(raw: str) -> str:
    s = raw.strip()
    m = re.search(r"\(([A-Za-z0-9^.-]{1,20})\)", s)
    return (m.group(1) if m else s.split()[0]).upper()


def _closing_prices(df: pd.DataFrame) -> pd.Series:
    """``Close`` may be a DataFrame under multi-column yfinance frames."""
    raw = df["Close"]
    if isinstance(raw, pd.DataFrame):
        raw = raw.iloc[:, 0] if raw.shape[1] > 0 else pd.Series(dtype="float64")
    return pd.to_numeric(raw, errors="coerce").dropna()


@tool
async def snapshot_quote(run_context: RunContext, ticker: str):
    t = _symbol(ticker)
    stock = yf.Ticker(t)
    info = await asyncio.to_thread(getattr, stock, "info")
    if not isinstance(info, dict):
        err = {"error": "quote unavailable", "ticker": t}
        run_context.session_state["last_quote"] = err
        return PortfolioHealthContent(last_quote=err)
    slim = {
        k: info.get(k)
        for k in ("symbol", "shortName", "regularMarketPrice", "marketCap", "trailingPE", "52WeekHigh", "52WeekLow")
        if info.get(k) is not None
    }
    run_context.session_state["last_quote"] = slim
    if slim:
        await maybe_embed_runtime_kb_payload(
            kb_name=f"runtime:yfinance:{t}:portfolio:snapshot_quote",
            description=f"Yahoo Finance quote snapshot for {t}",
            payload={"last_quote": slim},
            metadata={
                "source": "yfinance",
                "agent": "portfolio_health",
                "kind": "snapshot_quote",
                "ticker": t,
            },
        )
    return PortfolioHealthContent(last_quote=slim)


@tool
async def period_total_return(run_context: RunContext, ticker: str, period: str = "1y"):
    t = _symbol(ticker)
    data = await asyncio.to_thread(yf.download, t, period=period, interval="1d", progress=False)
    if data is None or data.empty:
        err = {"error": "no data", "ticker": t}
        run_context.session_state["last_return"] = err
        return PortfolioHealthContent(last_return=err)
    close = _closing_prices(data)
    if close.empty:
        err = {"error": "no close", "ticker": t}
        run_context.session_state["last_return"] = err
        return PortfolioHealthContent(last_return=err)
    first, last = float(close.iloc[0]), float(close.iloc[-1])
    ret = (last / first - 1.0) * 100.0 if first else 0.0
    out = {"ticker": t, "period": period, "total_return_pct": round(ret, 4)}
    run_context.session_state["last_return"] = out
    await maybe_embed_runtime_kb_payload(
        kb_name=f"runtime:yfinance:{t}:portfolio:period_total_return:{period}",
        description=f"Yahoo Finance OHLCV + total return {t} ({period})",
        payload={"summary": out, "ohlcv": frame_to_records_dict(data)},
        metadata={
            "source": "yfinance",
            "agent": "portfolio_health",
            "kind": "period_total_return",
            "ticker": t,
            "period": period,
        },
    )
    return PortfolioHealthContent(last_return=out)


@tool
async def compare_two_returns(run_context: RunContext, ticker_a: str, ticker_b: str, period: str = "1y"):
    sym_a, sym_b = _symbol(ticker_a), _symbol(ticker_b)

    def total_ret_pct(data: pd.DataFrame) -> float | None:
        if data is None or data.empty:
            return None
        c = _closing_prices(data)
        if c.empty:
            return None
        a0, b0 = float(c.iloc[0]), float(c.iloc[-1])
        return (b0 / a0 - 1.0) * 100.0 if a0 else None

    data_a = await asyncio.to_thread(yf.download, sym_a, period=period, interval="1d", progress=False)
    data_b = await asyncio.to_thread(yf.download, sym_b, period=period, interval="1d", progress=False)
    ra, rb = total_ret_pct(data_a), total_ret_pct(data_b)
    out = {
        "a": sym_a,
        "b": sym_b,
        "return_pct_a": None if ra is None else round(ra, 4),
        "return_pct_b": None if rb is None else round(rb, 4),
    }
    run_context.session_state["compare_returns"] = out
    if (data_a is not None and not data_a.empty) or (data_b is not None and not data_b.empty):
        payload = {
            "compare_returns": out,
            "ohlcv_a": frame_to_records_dict(data_a) if data_a is not None and not data_a.empty else None,
            "ohlcv_b": frame_to_records_dict(data_b) if data_b is not None and not data_b.empty else None,
        }
        await maybe_embed_runtime_kb_payload(
            kb_name=f"runtime:yfinance:{sym_a}_{sym_b}:portfolio:compare_returns:{period}",
            description=f"Yahoo Finance compare returns {sym_a} vs {sym_b} ({period})",
            payload=payload,
            metadata={
                "source": "yfinance",
                "agent": "portfolio_health",
                "kind": "compare_two_returns",
                "ticker_a": sym_a,
                "ticker_b": sym_b,
                "period": period,
            },
        )
    return PortfolioHealthContent(compare_returns=out)


def portfolio_health_tools_factory(agent=None, run_context=None):
    return [snapshot_quote, period_total_return, compare_two_returns]


async def merge_portfolio_health_output_from_session_state(run_output, run_context, **_kwargs):
    state = run_context.session_state or {}
    run_output.content = PortfolioHealthAgentOutput(
        portfolio_health=PortfolioHealthContent(
            last_quote=state.get("last_quote"),
            last_return=state.get("last_return"),
            compare_returns=state.get("compare_returns"),
        )
    )


def create_portfolio_health_agent(team=None, run_context=None):
    return Agent(
        db=db,
        knowledge=knowledge_base,
        search_knowledge=False,
        add_knowledge_to_context=False,
        name="portfolio_health_agent",
        description="Quick quote snapshots and simple total-return checks for holdings.",
        instructions="Retrieve live quote snapshots and compute total returns or comparisons for the requested tickers, then summarize the results without offering buy/sell recommendations.",
        reasoning=True,
        tools=portfolio_health_tools_factory,
        add_history_to_context=False,
        add_session_state_to_context=True,
        session_state={"last_quote": None, "last_return": None, "compare_returns": None},
        output_schema=PortfolioHealthAgentOutput,
        post_hooks=[merge_portfolio_health_output_from_session_state],
        model=OpenAIResponses(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
    )


portfolio_health_agent = create_portfolio_health_agent()
