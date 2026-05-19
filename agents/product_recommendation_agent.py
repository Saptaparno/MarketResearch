import os
import sys
from pathlib import Path

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.run import RunContext
from agno.tools import tool
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from agents.market_research_agent import db, knowledge_base
from models.schema import (
    FeeCompareIllustration,
    GoalMapContent,
    ProductRecommendationAgentOutput,
    ProductRecommendationContent,
    RankedSymbolEntry,
)

load_dotenv()


@tool
async def rank_symbol_list(
    run_context: RunContext, tickers_csv: str, risk_tolerance_1_to_5: int = 3
):
    """Lightweight ordering hint only; not a suitability or endorsement."""
    names = [x.strip().upper() for x in tickers_csv.split(",") if x.strip()]
    r = max(1, min(5, risk_tolerance_1_to_5))
    ranked = [
        {"symbol": n, "exploratory_rank": i + 1, "rough_fit_weight": round(0.15 * r + 0.05 * (len(names) - i), 2)}
        for i, n in enumerate(names)
    ]
    run_context.session_state["last_ranked"] = ranked
    return ProductRecommendationContent(last_ranked=ranked)


@tool
async def goal_to_categories(run_context: RunContext, goal: str, time_horizon_years: float = 5):
    g = goal.lower()
    cats: list[str] = []
    if any(k in g for k in ("income", "dividend", "yield")):
        cats.append("Dividend-oriented equities / core bonds (high-level)")
    if any(k in g for k in ("growth", "appreciate", "accumulate")):
        cats.append("Broad equity indices / diversified growth (high-level)")
    if time_horizon_years < 3:
        cats.append("Shorter-duration / capital preservation focus")
    if not cats:
        cats.append("Broad market index concepts as a generic discussion starting point")
    out = {"goal": goal.strip(), "time_horizon_years": time_horizon_years, "categories": cats}
    run_context.session_state["last_goal_map"] = out
    return ProductRecommendationContent(last_goal_map=out)


@tool
async def fee_drag_compare(
    run_context: RunContext,
    annual_fee_pct_a: float,
    annual_fee_pct_b: float,
    years: float = 10,
    assumed_annual_growth_pct: float = 6,
):
    """Illustrative ending balance on $100 with simple fee + growth assumptions."""
    if years <= 0:
        err = {"error": "years must be positive"}
        run_context.session_state["last_fee_compare"] = err
        return ProductRecommendationContent(last_fee_compare=err)
    r = assumed_annual_growth_pct / 100.0
    fa, fb = annual_fee_pct_a / 100.0, annual_fee_pct_b / 100.0

    def ending(fee: float) -> float:
        # single-period-style compounding approximation
        return 100.0 * ((1.0 + max(-0.99, r - fee)) ** years)

    out = {
        "starting_notional": 100,
        "years": years,
        "assumed_annual_growth_pct": assumed_annual_growth_pct,
        "fee_pct_a": annual_fee_pct_a,
        "fee_pct_b": annual_fee_pct_b,
        "ending_a": round(ending(fa), 2),
        "ending_b": round(ending(fb), 2),
    }
    run_context.session_state["last_fee_compare"] = out
    return ProductRecommendationContent(last_fee_compare=out)


def product_recommendation_tools_factory(agent=None, run_context=None):
    return [rank_symbol_list, goal_to_categories, fee_drag_compare]


def _coerce_ranked(entries):
    ranked = entries or []
    out: list[RankedSymbolEntry] = []
    for e in ranked:
        if isinstance(e, RankedSymbolEntry):
            out.append(e)
        else:
            out.append(RankedSymbolEntry.model_validate(e))
    return out


def _coerce_goal_map(raw):
    if raw is None:
        return None
    if isinstance(raw, GoalMapContent):
        return raw
    return GoalMapContent.model_validate(raw)


def _coerce_fee_compare(raw):
    if raw is None:
        return None
    if isinstance(raw, FeeCompareIllustration):
        return raw
    return FeeCompareIllustration.model_validate(raw)


async def merge_product_recommendation_output_from_session_state(run_output, run_context, **_kwargs):
    state = run_context.session_state or {}
    # Preserve text_response from the LLM's structured output before overwriting.
    # When no tool is called the only content is in text_response; without this it is lost.
    llm_content = run_output.content
    text_response = None
    if isinstance(llm_content, ProductRecommendationAgentOutput):
        text_response = llm_content.product_recommendation.text_response
    run_output.content = ProductRecommendationAgentOutput(
        product_recommendation=ProductRecommendationContent(
            last_ranked=_coerce_ranked(state.get("last_ranked")),
            last_goal_map=_coerce_goal_map(state.get("last_goal_map")),
            last_fee_compare=_coerce_fee_compare(state.get("last_fee_compare")),
            text_response=text_response,
        )
    )


def create_product_recommendation_agent(team=None, run_context=None):
    return Agent(
        db=db,
        knowledge=knowledge_base,
        search_knowledge=True,
        add_knowledge_to_context=True,
        name="product_recommendation_agent",
        instructions=(
            "Map user goals to broad asset categories, rank symbol lists, or illustrate fee drag "
            "using the available tools — always call goal_to_categories for goal-mapping requests "
            "and rank_symbol_list when tickers are provided. "
            "Always populate text_response with a concise narrative explanation that: "
            "(1) describes the suggested broad asset categories in plain English, "
            "(2) frames all suggestions as exploratory discussion points, not personalized advice, "
            "(3) includes a caveat that this is not a personalized recommendation."
        ),
        description=(
            "Exploratory product discussion only: rank lists, map goals to broad categories, "
            "and illustrate fee drag. Not personalized advice or an offer of any product."
        ),
        reasoning=True,
        tools=product_recommendation_tools_factory,
        add_history_to_context=True,
        add_session_state_to_context=True,
        session_state={"last_ranked": [], "last_goal_map": None, "last_fee_compare": None},
        output_schema=ProductRecommendationAgentOutput,
        post_hooks=[merge_product_recommendation_output_from_session_state],
        model=OpenAIResponses(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
    )


product_recommendation_agent = create_product_recommendation_agent()
