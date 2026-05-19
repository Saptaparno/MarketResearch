import asyncio
import sys
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agno.eval.reliability import ReliabilityEval, ReliabilityResult
from agno.run.agent import RunOutput
from agno.workflow import StepOutput
from dotenv import load_dotenv

from agents.financial_planning_agent import create_financial_planning_agent
from agents.general_enquiry_agent import create_general_enquiry_agent
from agents.portfolio_health_agent import create_portfolio_health_agent
from agents.risk_analyst_agent import create_risk_analyst_agent
from agents.trade_agent import create_trade_agent

load_dotenv()


# ---------------------------------------------------------------------------
# Original reliability checks (individual agent tool calls)
# ---------------------------------------------------------------------------

async def eval_financial_planning_calls_fv_tool():
    agent = create_financial_planning_agent()
    response: RunOutput = await agent.arun("What is the future value of $5,000 at 8% for 5 years?")
    result: Optional[ReliabilityResult] = await ReliabilityEval(
        name="Financial Planning — future_value_lump_sum tool call",
        agent_response=response,
        expected_tool_calls=["future_value_lump_sum"],
    ).arun(print_results=True)
    assert result is not None
    result.assert_passed()


async def eval_financial_planning_calls_monthly_tool():
    agent = create_financial_planning_agent()
    response: RunOutput = await agent.arun("How much will I have if I save $300/month for 3 years at 5% annual rate?")
    result: Optional[ReliabilityResult] = await ReliabilityEval(
        name="Financial Planning — future_value_monthly_contrib tool call",
        agent_response=response,
        expected_tool_calls=["future_value_monthly_contrib"],
    ).arun(print_results=True)
    assert result is not None
    result.assert_passed()


async def eval_financial_planning_calls_loan_tool():
    agent = create_financial_planning_agent()
    response: RunOutput = await agent.arun("What is the monthly payment on a $200,000 mortgage at 6% over 30 years?")
    result: Optional[ReliabilityResult] = await ReliabilityEval(
        name="Financial Planning — loan_payment_monthly tool call",
        agent_response=response,
        expected_tool_calls=["loan_payment_monthly"],
    ).arun(print_results=True)
    assert result is not None
    result.assert_passed()


async def eval_portfolio_health_calls_snapshot_quote():
    agent = create_portfolio_health_agent()
    response: RunOutput = await agent.arun("What is the current price and market cap of AAPL?")
    result: Optional[ReliabilityResult] = await ReliabilityEval(
        name="Portfolio Health — snapshot_quote tool call",
        agent_response=response,
        expected_tool_calls=["snapshot_quote"],
    ).arun(print_results=True)
    assert result is not None
    result.assert_passed()


async def eval_portfolio_health_calls_period_return():
    agent = create_portfolio_health_agent()
    response: RunOutput = await agent.arun("What is the 1-year total return for MSFT?")
    result: Optional[ReliabilityResult] = await ReliabilityEval(
        name="Portfolio Health — period_total_return tool call",
        agent_response=response,
        expected_tool_calls=["period_total_return"],
    ).arun(print_results=True)
    assert result is not None
    result.assert_passed()


async def eval_risk_analyst_calls_volatility():
    agent = create_risk_analyst_agent()
    response: RunOutput = await agent.arun("Calculate the annualized volatility of SPY over the last year.")
    result: Optional[ReliabilityResult] = await ReliabilityEval(
        name="Risk Analyst — daily_return_volatility tool call",
        agent_response=response,
        expected_tool_calls=["daily_return_volatility"],
    ).arun(print_results=True)
    assert result is not None
    result.assert_passed()


async def eval_risk_analyst_calls_drawdown():
    agent = create_risk_analyst_agent()
    response: RunOutput = await agent.arun("What is the maximum drawdown for TSLA over the past year?")
    result: Optional[ReliabilityResult] = await ReliabilityEval(
        name="Risk Analyst — max_drawdown_pct tool call",
        agent_response=response,
        expected_tool_calls=["max_drawdown_pct"],
    ).arun(print_results=True)
    assert result is not None
    result.assert_passed()


async def eval_trade_agent_calls_propose_trade():
    agent = create_trade_agent()
    response: RunOutput = await agent.arun("Propose a buy of 10 shares of AAPL.")
    result: Optional[ReliabilityResult] = await ReliabilityEval(
        name="Trade Agent — propose_trade tool call",
        agent_response=response,
        expected_tool_calls=["propose_trade"],
    ).arun(print_results=True)
    assert result is not None
    result.assert_passed()


async def eval_general_enquiry_calls_normalize():
    agent = create_general_enquiry_agent()
    response: RunOutput = await agent.arun("Normalize this question:  what  is   ETF  ?")
    result: Optional[ReliabilityResult] = await ReliabilityEval(
        name="General Enquiry — normalize_question tool call",
        agent_response=response,
        expected_tool_calls=["normalize_question"],
    ).arun(print_results=True)
    assert result is not None
    result.assert_passed()


# ---------------------------------------------------------------------------
# Morning briefing workflow reliability checks (Phase 10 spec)
# ---------------------------------------------------------------------------

async def eval_morning_briefing_fetch_portfolio_once():
    """fetch_portfolio must appear exactly once in the morning briefing workflow.

    The spec requires the morning briefing to call fetch_portfolio exactly once
    per run. This test inspects the workflow's step graph to assert there is no
    duplication — the Parallel block wraps it once and no other step calls it.
    """
    from core.workflow import morning_briefing_workflow, fetch_portfolio

    fetch_count = 0
    for step in morning_briefing_workflow.steps:
        # Parallel blocks expose their sub-steps via .steps
        sub_steps = getattr(step, "steps", None)
        if sub_steps is not None:
            for sub in sub_steps:
                if sub is fetch_portfolio:
                    fetch_count += 1
        elif step is fetch_portfolio:
            fetch_count += 1

    assert fetch_count == 1, (
        f"Expected fetch_portfolio to appear exactly once in morning_briefing_workflow, "
        f"found {fetch_count}"
    )
    print(f"  fetch_portfolio count in workflow: {fetch_count} ✓")


async def eval_portfolio_drop_sets_red_flag():
    """fetch_portfolio must set has_red_flag=True when any position dropped > 5%.

    Mocks portfolio_health_agent.arun to return a -7 % 1-year return for AAPL,
    then asserts the step correctly sets has_red_flag and flagged_tickers.
    """
    from core.workflow import fetch_portfolio

    mock_resp = MagicMock()
    mock_resp.content.portfolio_health.last_return = {"total_return_pct": -7.0}

    session_state: dict = {"portfolio_tickers": ["AAPL"]}

    with patch("core.workflow.portfolio_health_agent") as mock_agent:
        mock_agent.arun = AsyncMock(return_value=mock_resp)
        result: StepOutput = await fetch_portfolio(None, session_state)

    assert result.success is True, "fetch_portfolio step should succeed"
    assert session_state.get("has_red_flag") is True, (
        "has_red_flag should be True when a position dropped > 5%"
    )
    assert "AAPL" in session_state.get("flagged_tickers", []), (
        "AAPL should be in flagged_tickers after a -7% return"
    )
    print("  has_red_flag=True correctly set for -7% return ✓")


async def eval_risk_agent_triggered_on_drop():
    """red_flag_evaluator must return True only when has_red_flag is set in session_state.

    The Condition(steps=[risk_deep_dive], evaluator=red_flag_evaluator) in the
    morning briefing workflow fires risk_deep_dive (which calls risk_analyst_agent)
    when the evaluator returns True. This test validates the evaluator gate.
    """
    from core.workflow import red_flag_evaluator

    assert await red_flag_evaluator(None, {"has_red_flag": True}) is True, (
        "red_flag_evaluator should return True when has_red_flag=True"
    )
    assert await red_flag_evaluator(None, {"has_red_flag": False}) is False, (
        "red_flag_evaluator should return False when has_red_flag=False"
    )
    assert await red_flag_evaluator(None, {}) is False, (
        "red_flag_evaluator should return False when has_red_flag is absent"
    )
    print("  red_flag_evaluator gate behaviour confirmed ✓")


if __name__ == "__main__":
    asyncio.run(eval_financial_planning_calls_fv_tool())
    asyncio.run(eval_financial_planning_calls_monthly_tool())
    asyncio.run(eval_financial_planning_calls_loan_tool())
    asyncio.run(eval_portfolio_health_calls_snapshot_quote())
    asyncio.run(eval_portfolio_health_calls_period_return())
    asyncio.run(eval_risk_analyst_calls_volatility())
    asyncio.run(eval_risk_analyst_calls_drawdown())
    asyncio.run(eval_trade_agent_calls_propose_trade())
    asyncio.run(eval_general_enquiry_calls_normalize())
    asyncio.run(eval_morning_briefing_fetch_portfolio_once())
    asyncio.run(eval_portfolio_drop_sets_red_flag())
    asyncio.run(eval_risk_agent_triggered_on_drop())
