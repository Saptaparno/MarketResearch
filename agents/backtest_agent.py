import asyncio
import os
import re
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.run import RunContext
from agno.run.cancel import is_cancelled
from agno.tools import tool
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from agents.market_research_agent import db, frame_to_records_dict

load_dotenv()

_VALID_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}

_PERIOD_MAP = {
    "last year": "1y", "one year": "1y", "past year": "1y",
    "last month": "1mo", "one month": "1mo", "past month": "1mo",
    "year to date": "ytd",
}


def _normalize_range(range_str: str) -> str:
    """Map natural-language period strings to valid yfinance period codes."""
    key = range_str.strip().lower()
    if key in _VALID_PERIODS:
        return key
    if key in _PERIOD_MAP:
        return _PERIOD_MAP[key]
    m = re.match(r"(\d+)\s*(years?|yr|months?|mo|days?|d)\b", key)
    if m:
        n, unit = m.group(1), m.group(2)
        if unit.startswith("y"):
            return f"{n}y"
        if unit.startswith("mo") or unit.startswith("m"):
            return f"{n}mo"
        if unit.startswith("d"):
            return f"{n}d"
    return "1y"


def _tickers(strategy: str) -> list[str]:
    return [t.strip().upper() for t in strategy.split(",") if t.strip()]


def _sharpe(returns: pd.Series, risk_free: float = 0.05) -> float:
    excess = returns - risk_free / 252
    return float((excess.mean() / excess.std()) * (252 ** 0.5)) if excess.std() > 0 else 0.0


def _max_drawdown(close: pd.Series) -> float:
    peak = close.cummax()
    return float(((close - peak) / peak).min() * 100)


@tool
async def backtest(run_context: RunContext, strategy: str, range: str = "1y") -> dict:
    """
    Runs a long-running backtest for the given strategy (comma-separated tickers)
    over the specified date range. Returns per-ticker metrics and a portfolio summary.
    Supports mid-run cancellation.
    """
    tickers = _tickers(strategy)
    if not tickers:
        return {"error": "No tickers provided in strategy."}

    period = _normalize_range(range)
    results: dict = {}

    for ticker in tickers:
        if is_cancelled(run_context.run_id):
            return {"status": "cancelled", "completed_tickers": list(results.keys())}

        data = await asyncio.to_thread(
            yf.download, ticker, period=period, interval="1d", progress=False
        )
        if data is None or data.empty:
            results[ticker] = {"error": "no data"}
            continue

        # simulate processing latency so the run is genuinely long-running
        await asyncio.sleep(2)

        if is_cancelled(run_context.run_id):
            return {"status": "cancelled", "completed_tickers": list(results.keys())}

        raw_close = data["Close"]
        close: pd.Series = (
            raw_close.iloc[:, 0] if isinstance(raw_close, pd.DataFrame) else raw_close
        )
        close = pd.to_numeric(close, errors="coerce").dropna()
        returns = close.pct_change().dropna()

        total_return = float((close.iloc[-1] / close.iloc[0] - 1) * 100) if len(close) > 1 else 0.0
        annualized_vol = float(returns.std() * (252 ** 0.5) * 100) if len(returns) > 0 else 0.0

        results[ticker] = {
            "total_return_pct": round(total_return, 4),
            "annualized_vol_pct": round(annualized_vol, 4),
            "sharpe_ratio": round(_sharpe(returns), 4),
            "max_drawdown_pct": round(_max_drawdown(close), 4),
            "data_points": len(close),
        }

    if is_cancelled(run_context.run_id):
        return {"status": "cancelled", "completed_tickers": list(results.keys())}

    valid_returns = [v["total_return_pct"] for v in results.values() if "total_return_pct" in v]
    avg_return = sum(valid_returns) / len(valid_returns) if valid_returns else 0.0

    return {
        "status": "completed",
        "strategy": strategy,
        "range": period,
        "per_ticker": results,
        "portfolio_avg_return_pct": round(avg_return, 4),
    }


def create_backtest_agent(team=None, run_context=None) -> Agent:
    return Agent(
        db=db,
        name="backtest_agent",
        description="Runs long-running strategy backtests with per-ticker return, volatility, Sharpe, and drawdown metrics. Supports background execution and mid-run cancellation.",
        instructions=(
            "When asked to backtest a strategy, call the backtest tool with the comma-separated "
            "tickers and the date range. Reply with a brief summary and a compact Markdown table "
            "of per-ticker metrics only. Keep the response under 300 words. Always end with: "
            "'Past performance is not indicative of future results.'"
        ),
        tools=[backtest],
        add_history_to_context=False,
        add_session_state_to_context=False,
        model=OpenAIResponses(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), max_output_tokens=4096),
    )


backtest_agent = create_backtest_agent()
