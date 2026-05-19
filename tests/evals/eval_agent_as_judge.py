import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agno.agent import Agent
from agno.eval.agent_as_judge import AgentAsJudgeEval
from agno.models.openai import OpenAIResponses
from agno.run.agent import RunOutput
from dotenv import load_dotenv

from agents.financial_planning_agent import create_financial_planning_agent
from agents.general_enquiry_agent import create_general_enquiry_agent
from agents.market_research_agent import create_market_research_agent
from agents.product_recommendation_agent import create_product_recommendation_agent
from agents.risk_analyst_agent import create_risk_analyst_agent
from models.schema import GoalMapContent

load_dotenv()


# ---------------------------------------------------------------------------
# agno_evals persistence helper
# ---------------------------------------------------------------------------

async def _save_to_agno_evals(
    eval_name: str,
    score: float | None,
    passed: bool | None,
    criteria: str,
    input_text: str,
    output_text: str,
) -> None:
    """Persist a judge-eval result to the agno_evals table.

    Gracefully skips if DATABASE_URL is not set or the DB is unreachable,
    so the suite still passes in environments without Postgres.
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("  [agno_evals] DATABASE_URL not set — skipping persistence.")
        return
    try:
        import asyncpg

        # Normalise URL scheme for asyncpg (it needs postgresql://)
        url = db_url
        for prefix, replacement in (
            ("postgresql+asyncpg://", "postgresql://"),
            ("postgres+asyncpg://", "postgresql://"),
            ("postgres://", "postgresql://"),
        ):
            if url.startswith(prefix):
                url = replacement + url[len(prefix):]
                break

        conn = await asyncpg.connect(url)
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agno_evals (
                id         BIGSERIAL PRIMARY KEY,
                eval_name  TEXT        NOT NULL,
                score      FLOAT,
                passed     BOOLEAN,
                criteria   TEXT,
                input      TEXT,
                output     TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
        await conn.execute(
            "INSERT INTO agno_evals (eval_name, score, passed, criteria, input, output) "
            "VALUES ($1, $2, $3, $4, $5, $6)",
            eval_name,
            score,
            passed,
            criteria[:1000],
            input_text[:2000],
            output_text[:2000],
        )
        await conn.close()
        print(f"  [agno_evals] Saved result for '{eval_name}'.")
    except Exception as exc:
        print(f"  [agno_evals] Save skipped ({type(exc).__name__}: {exc})")


# ---------------------------------------------------------------------------
# Reporter agent (Phase 10: grade Reporter on evidence-cited/balanced/actionable)
# ---------------------------------------------------------------------------

def _create_reporter_agent() -> Agent:
    """Minimal Reporter agent used exclusively in judge evals."""
    return Agent(
        name="reporter",
        description="Morning briefing reporter for Valura.",
        instructions=[
            "You are Valura's morning briefing reporter.",
            "When given portfolio, watchlist, and macro data, produce a Markdown briefing.",
            "Always: (1) cite specific numeric data and sources, "
            "(2) present both bullish and bearish perspectives, "
            "(3) end with a 'Key Actions' bullet list.",
        ],
        model=OpenAIResponses(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
    )


_REPORTER_INPUT = (
    "Write a morning briefing for this portfolio snapshot:\n"
    "- NVDA: +42 % YTD, PE 65, 52-week high $974\n"
    "- AAPL: +18 % YTD, PE 28, recent revenue miss\n"
    "- BND:  -2 % YTD, yield 4.1 %\n"
    "SPY annualised vol: 14.8 %. No circuit-breaker. Watchlist: MSFT, META."
)


async def judge_reporter_evidence_cited():
    """Reporter briefing must cite specific numeric evidence from the input data."""
    agent = _create_reporter_agent()
    response: RunOutput = await agent.arun(_REPORTER_INPUT)
    output_text = str(response.content)

    criteria = (
        "The response must cite at least two specific numeric figures from the input data "
        "(e.g. percentage returns, PE ratios, volatility values, or price levels). "
        "Generic statements without numbers do not count."
    )
    result = await AgentAsJudgeEval(
        name="Reporter — evidence-cited",
        criteria=criteria,
        scoring_strategy="numeric",
        threshold=7,
    ).arun(input=_REPORTER_INPUT, output=output_text, print_results=True)

    score = getattr(result, "score", None) if result else None
    passed = score is not None and score >= 7
    await _save_to_agno_evals(
        eval_name="reporter_evidence_cited",
        score=score,
        passed=passed,
        criteria=criteria,
        input_text=_REPORTER_INPUT,
        output_text=output_text,
    )
    assert passed, f"Reporter evidence-cited score {score} < threshold 7"


async def judge_reporter_balanced():
    """Reporter briefing must present both bullish and bearish perspectives."""
    agent = _create_reporter_agent()
    response: RunOutput = await agent.arun(_REPORTER_INPUT)
    output_text = str(response.content)

    criteria = (
        "The response must present at least one bullish point (positive outlook or upside) "
        "AND at least one bearish point (risk, downside, or concern) for the portfolio. "
        "A one-sided briefing that only praises or only warns does not meet the bar."
    )
    result = await AgentAsJudgeEval(
        name="Reporter — balanced",
        criteria=criteria,
        scoring_strategy="numeric",
        threshold=7,
    ).arun(input=_REPORTER_INPUT, output=output_text, print_results=True)

    score = getattr(result, "score", None) if result else None
    passed = score is not None and score >= 7
    await _save_to_agno_evals(
        eval_name="reporter_balanced",
        score=score,
        passed=passed,
        criteria=criteria,
        input_text=_REPORTER_INPUT,
        output_text=output_text,
    )
    assert passed, f"Reporter balanced score {score} < threshold 7"


async def judge_reporter_actionable():
    """Reporter briefing must end with a concrete Key Actions bullet list."""
    agent = _create_reporter_agent()
    response: RunOutput = await agent.arun(_REPORTER_INPUT)
    output_text = str(response.content)

    criteria = (
        "The response must end with a clearly labelled 'Key Actions' (or similar) "
        "section containing at least two bullet points. Each bullet must describe a "
        "specific, concrete action an analyst could consider — not vague platitudes."
    )
    result = await AgentAsJudgeEval(
        name="Reporter — actionable",
        criteria=criteria,
        scoring_strategy="numeric",
        threshold=7,
    ).arun(input=_REPORTER_INPUT, output=output_text, print_results=True)

    score = getattr(result, "score", None) if result else None
    passed = score is not None and score >= 7
    await _save_to_agno_evals(
        eval_name="reporter_actionable",
        score=score,
        passed=passed,
        criteria=criteria,
        input_text=_REPORTER_INPUT,
        output_text=output_text,
    )
    assert passed, f"Reporter actionable score {score} < threshold 7"


# ---------------------------------------------------------------------------
# Original agent-as-judge checks
# ---------------------------------------------------------------------------

async def judge_market_research_structure():
    agent = create_market_research_agent()
    response: RunOutput = await agent.arun("Give me a brief overview of AAPL: current price, PE ratio, and 52-week range.")
    await AgentAsJudgeEval(
        name="Market Research — structured data response",
        criteria=(
            "The response must include numerical data for at least two of: current price, PE ratio, or 52-week range. "
            "Data should be sourced from real market data, not fabricated."
        ),
        scoring_strategy="numeric",
        threshold=7,
    ).arun(input="Brief overview of AAPL", output=str(response.content), print_results=True)


async def judge_risk_analyst_disclaimer():
    agent = create_risk_analyst_agent()
    response: RunOutput = await agent.arun("What is the annualized volatility and max drawdown of NVDA over the past year?")
    await AgentAsJudgeEval(
        name="Risk Analyst — informational disclaimer present",
        criteria=(
            "The response must present numeric risk metrics (volatility or drawdown percentage) "
            "and include a disclaimer that the information is for informational purposes only."
        ),
        scoring_strategy="numeric",
        threshold=7,
    ).arun(input="Volatility and max drawdown of NVDA", output=str(response.content), print_results=True)


async def judge_financial_planning_educational_tone():
    agent = create_financial_planning_agent()
    response: RunOutput = await agent.arun("What monthly payment would I have on a $250,000 mortgage at 7% over 30 years?")
    await AgentAsJudgeEval(
        name="Financial Planning — educational tone and no personalized advice",
        criteria=(
            "The response must include the calculated monthly payment amount, present it clearly, "
            "and must NOT give personalized investment or financial advice."
        ),
        scoring_strategy="numeric",
        threshold=7,
    ).arun(
        input="Monthly mortgage payment on $250k at 7% over 30 years",
        output=str(response.content),
        print_results=True,
    )


async def judge_general_enquiry_concise_answer():
    agent = create_general_enquiry_agent()
    response: RunOutput = await agent.arun("What is dollar-cost averaging?")
    # Extract text_response; fall back to full repr so the eval is never silently empty.
    content = response.content
    output_text = (
        content.general_enquiry.text_response
        if hasattr(content, "general_enquiry") and content.general_enquiry.text_response
        else str(content)
    )
    assert output_text and output_text != str(type(content)), (
        "general_enquiry.text_response is empty — the LLM did not populate it. "
        "Check agent instructions and post-hook."
    )
    await AgentAsJudgeEval(
        name="General Enquiry — clear and concise finance explanation",
        criteria=(
            "The response must explain dollar-cost averaging in plain English, "
            "be concise (under 200 words), and avoid jargon without definition."
        ),
        scoring_strategy="numeric",
        threshold=7,
    ).arun(input="What is dollar-cost averaging?", output=output_text, print_results=True)


async def judge_product_recommendation_exploratory_framing():
    agent = create_product_recommendation_agent()
    response: RunOutput = await agent.arun(
        "Map my goal of long-term growth with a 10-year horizon to broad asset categories."
    )
    content = response.content

    # Prefer text_response (narrative); fall back to composing from tool-call data so
    # a successful goal_to_categories call is not wasted even if text_response is empty.
    output_text = None
    if hasattr(content, "product_recommendation"):
        rec = content.product_recommendation
        output_text = rec.text_response
        if not output_text and rec.last_goal_map:
            gm = rec.last_goal_map
            cats = gm.categories if isinstance(gm, GoalMapContent) else gm.get("categories", [])
            output_text = (
                f"Goal: {gm.goal if isinstance(gm, GoalMapContent) else gm.get('goal', '')}. "
                f"Time horizon: {gm.time_horizon_years if isinstance(gm, GoalMapContent) else gm.get('time_horizon_years', '')} years. "
                f"Suggested broad categories (exploratory, not personalized advice): {', '.join(cats)}."
            )
    output_text = output_text or str(content)

    assert output_text and "ProductRecommendationContent" not in output_text or (
        hasattr(content, "product_recommendation") and
        content.product_recommendation.last_goal_map is not None
    ), (
        "product_recommendation output is empty — neither text_response nor goal_to_categories "
        "tool call produced usable content. Check agent instructions and post-hook."
    )

    await AgentAsJudgeEval(
        name="Product Recommendation — exploratory framing, no specific product endorsement",
        criteria=(
            "The response must suggest broad asset categories (not specific tickers or funds), "
            "frame suggestions as exploratory rather than personalized advice, "
            "and include a caveat that this is not a personalized recommendation."
        ),
        scoring_strategy="numeric",
        threshold=7,
    ).arun(
        input="Long-term growth goal mapping, 10-year horizon",
        output=output_text,
        print_results=True,
    )


if __name__ == "__main__":
    asyncio.run(judge_reporter_evidence_cited())
    asyncio.run(judge_reporter_balanced())
    asyncio.run(judge_reporter_actionable())
    asyncio.run(judge_market_research_structure())
    asyncio.run(judge_risk_analyst_disclaimer())
    asyncio.run(judge_financial_planning_educational_tone())
    asyncio.run(judge_general_enquiry_concise_answer())
    asyncio.run(judge_product_recommendation_exploratory_framing())
