import os
import sys
from pathlib import Path

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools import tool
from agno.run import RunContext
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from agents.market_research_agent import db, knowledge_base
from models.schema import FinancialPlanningAgentOutput, FinancialPlanningContent

load_dotenv()


@tool
async def future_value_lump_sum(run_context: RunContext, principal: float, years: float, annual_rate_pct: float):
    r = annual_rate_pct / 100.0
    fv = principal * (1.0 + r) ** years if years >= 0 else principal
    out = {"principal": principal, "years": years, "annual_rate_pct": annual_rate_pct, "fv": round(fv, 2)}
    run_context.session_state["last_fv_lump"] = out
    return FinancialPlanningContent(last_fv_lump=out)


@tool
async def future_value_monthly_contrib(
    run_context: RunContext, monthly: float, years: float, annual_rate_pct: float
):
    if years <= 0:
        err = {"error": "years must be positive"}
        run_context.session_state["last_fv_monthly"] = err
        return FinancialPlanningContent(last_fv_monthly=err)
    r = (annual_rate_pct / 100.0) / 12.0
    n = int(years * 12)
    if r == 0:
        fv = monthly * n
    else:
        fv = monthly * (((1.0 + r) ** n - 1.0) / r)
    out = {"monthly": monthly, "months": n, "annual_rate_pct": annual_rate_pct, "fv": round(fv, 2)}
    run_context.session_state["last_fv_monthly"] = out
    return FinancialPlanningContent(last_fv_monthly=out)


@tool
async def loan_payment_monthly(
    run_context: RunContext, loan_amount: float, years: float, annual_rate_pct: float
):
    if years <= 0 or loan_amount <= 0:
        err = {"error": "invalid loan or term"}
        run_context.session_state["last_payment"] = err
        return FinancialPlanningContent(last_payment=err)
    r = (annual_rate_pct / 100.0) / 12.0
    n = int(max(1, years * 12))
    if r == 0:
        pay = loan_amount / n
    else:
        pay = loan_amount * (r * (1.0 + r) ** n) / ((1.0 + r) ** n - 1.0)
    out = {
        "loan_amount": loan_amount,
        "months": n,
        "annual_rate_pct": annual_rate_pct,
        "payment": round(pay, 2),
    }
    run_context.session_state["last_payment"] = out
    return FinancialPlanningContent(last_payment=out)


def financial_planning_tools_factory(agent=None, run_context=None):
    return [future_value_lump_sum, future_value_monthly_contrib, loan_payment_monthly]


async def merge_financial_planning_output_from_session_state(run_output, run_context, **_kwargs):
    state = run_context.session_state or {}
    run_output.content = FinancialPlanningAgentOutput(
        financial_planning=FinancialPlanningContent(
            last_fv_lump=state.get("last_fv_lump"),
            last_fv_monthly=state.get("last_fv_monthly"),
            last_payment=state.get("last_payment"),
        )
    )


def create_financial_planning_agent(team=None, run_context=None, structured_output=True):
    kwargs = dict(
        db=db,
        knowledge=knowledge_base,
        search_knowledge=True,
        add_knowledge_to_context=True,
        name="financial_planning_agent",
        description="Simple FV and payment calculators (educational; not personalized advice).",
        instructions="Use the provided tools to calculate future values and loan payments, then present results clearly with a brief educational note; never give personalized financial advice.",
        reasoning=True,
        tools=financial_planning_tools_factory,
        add_history_to_context=True,
        add_session_state_to_context=True,
        session_state={"last_fv_lump": None, "last_fv_monthly": None, "last_payment": None},
        model=OpenAIResponses(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
    )
    if structured_output:
        kwargs["output_schema"] = FinancialPlanningAgentOutput
        kwargs["post_hooks"] = [merge_financial_planning_output_from_session_state]
    return Agent(**kwargs)


financial_planning_agent = create_financial_planning_agent()
