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
from models.schema import RiskAnalystAgentOutput, RiskAnalystContent

load_dotenv()


def _symbol(raw: str) -> str:
    s = raw.strip()
    m = re.search(r"\(([A-Za-z0-9^.-]{1,20})\)", s)
    return (m.group(1) if m else s.split()[0]).upper()


def _closing_prices(df: pd.DataFrame) -> pd.Series:
    raw = df["Close"]
    if isinstance(raw, pd.DataFrame):
        raw = raw.iloc[:, 0] if raw.shape[1] > 0 else pd.Series(dtype="float64")
    return pd.to_numeric(raw, errors="coerce").dropna()


def _daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()


@tool
async def daily_return_volatility(run_context: RunContext, ticker: str, period: str = "1y"):
    t = _symbol(ticker)
    data = await asyncio.to_thread(yf.download, t, period=period, interval="1d", progress=False)
    if data is None or data.empty:
        err = {"error": "no data", "ticker": t}
        run_context.session_state["last_vol"] = err
        return RiskAnalystContent(last_vol=err)
    r = _daily_returns(_closing_prices(data))
    if r.empty:
        err = {"error": "no returns", "ticker": t}
        run_context.session_state["last_vol"] = err
        return RiskAnalystContent(last_vol=err)
    out = {
        "ticker": t,
        "period": period,
        "annualized_vol_pct": round(float(r.std() * (252**0.5) * 100), 4),
    }
    run_context.session_state["last_vol"] = out
    await maybe_embed_runtime_kb_payload(
        kb_name=f"runtime:yfinance:{t}:risk:daily_volatility:{period}",
        description=f"Yahoo Finance OHLCV + annualized vol {t} ({period})",
        payload={"metrics": out, "ohlcv": frame_to_records_dict(data)},
        metadata={
            "source": "yfinance",
            "agent": "risk_analyst",
            "kind": "daily_return_volatility",
            "ticker": t,
            "period": period,
        },
    )
    return RiskAnalystContent(last_vol=out)


@tool
async def max_drawdown_pct(run_context: RunContext, ticker: str, period: str = "1y"):
    t = _symbol(ticker)
    data = await asyncio.to_thread(yf.download, t, period=period, interval="1d", progress=False)
    if data is None or data.empty:
        err = {"error": "no data", "ticker": t}
        run_context.session_state["last_dd"] = err
        return RiskAnalystContent(last_dd=err)
    close = _closing_prices(data)
    if close.empty:
        err = {"error": "no close", "ticker": t}
        run_context.session_state["last_dd"] = err
        return RiskAnalystContent(last_dd=err)
    peak = close.cummax()
    dd = ((close - peak) / peak).min() * 100.0
    out = {"ticker": t, "period": period, "max_drawdown_pct": round(float(dd), 4)}
    run_context.session_state["last_dd"] = out
    await maybe_embed_runtime_kb_payload(
        kb_name=f"runtime:yfinance:{t}:risk:max_drawdown:{period}",
        description=f"Yahoo Finance OHLCV + max drawdown {t} ({period})",
        payload={"metrics": out, "ohlcv": frame_to_records_dict(data)},
        metadata={
            "source": "yfinance",
            "agent": "risk_analyst",
            "kind": "max_drawdown_pct",
            "ticker": t,
            "period": period,
        },
    )
    return RiskAnalystContent(last_dd=out)


@tool
async def spy_return_correlation(run_context: RunContext, ticker: str, period: str = "1y"):
    t = _symbol(ticker)
    d1 = await asyncio.to_thread(yf.download, t, period=period, interval="1d", progress=False)
    d2 = await asyncio.to_thread(yf.download, "SPY", period=period, interval="1d", progress=False)
    if d1 is None or d1.empty or d2 is None or d2.empty:
        err = {"error": "no data", "ticker": t}
        run_context.session_state["last_corr"] = err
        return RiskAnalystContent(last_corr=err)
    c1 = _closing_prices(d1)
    c2 = _closing_prices(d2)
    r1 = _daily_returns(c1)
    r2 = _daily_returns(c2)
    joined = pd.concat([r1, r2], axis=1, join="inner").dropna()
    joined.columns = ["a", "b"]
    if len(joined) < 10:
        err = {"error": "too few points", "ticker": t}
        run_context.session_state["last_corr"] = err
        return RiskAnalystContent(last_corr=err)
    corr = round(float(joined["a"].corr(joined["b"])), 4)
    out = {"ticker": t, "period": period, "daily_return_corr_spy": corr}
    run_context.session_state["last_corr"] = out
    await maybe_embed_runtime_kb_payload(
        kb_name=f"runtime:yfinance:{t}:risk:spy_correlation:{period}",
        description=f"Yahoo Finance returns correlation {t} vs SPY ({period})",
        payload={
            "metrics": out,
            "ohlcv_ticker": frame_to_records_dict(d1),
            "ohlcv_spy": frame_to_records_dict(d2),
        },
        metadata={
            "source": "yfinance",
            "agent": "risk_analyst",
            "kind": "spy_return_correlation",
            "ticker": t,
            "period": period,
        },
    )
    return RiskAnalystContent(last_corr=out)


def risk_analyst_tools_factory(agent=None, run_context=None):
    return [daily_return_volatility, max_drawdown_pct, spy_return_correlation]


_DISCLAIMER = (
    "For informational purposes only. "
    "These metrics are not investment advice and should not be used as the sole basis "
    "for any investment decision."
)


async def merge_risk_analyst_output_from_session_state(run_output, run_context, **_kwargs):
    state = run_context.session_state or {}
    run_output.content = RiskAnalystAgentOutput(
        risk_analyst=RiskAnalystContent(
            last_vol=state.get("last_vol"),
            last_dd=state.get("last_dd"),
            last_corr=state.get("last_corr"),
            disclaimer=_DISCLAIMER,
        )
    )


def create_risk_analyst_agent(team=None, run_context=None):
    return Agent(
        db=db,
        knowledge=knowledge_base,
        search_knowledge=False,
        add_knowledge_to_context=False,
        name="risk_analyst_agent",
        description="Rough volatility, drawdown, and SPY correlation (not investment advice).",
        instructions="Compute annualized volatility, max drawdown, and SPY correlation for the requested ticker using the available tools, and present the metrics with a disclaimer that they are for informational purposes only.",
        reasoning=True,
        tools=risk_analyst_tools_factory,
        add_history_to_context=False,
        add_session_state_to_context=True,
        session_state={"last_vol": None, "last_dd": None, "last_corr": None},
        output_schema=RiskAnalystAgentOutput,
        post_hooks=[merge_risk_analyst_output_from_session_state],
        model=OpenAIResponses(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
    )


risk_analyst_agent = create_risk_analyst_agent()
