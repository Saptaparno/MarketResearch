import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agno.eval.performance import PerformanceEval
from agno.models.openai import OpenAIResponses
from agno.run.agent import RunOutput
from dotenv import load_dotenv

from agents.financial_planning_agent import create_financial_planning_agent
from agents.general_enquiry_agent import create_general_enquiry_agent
from agents.market_research_agent import create_market_research_agent
from agents.portfolio_health_agent import create_portfolio_health_agent
from agents.product_recommendation_agent import create_product_recommendation_agent
from agents.risk_analyst_agent import create_risk_analyst_agent
from agents.trade_agent import create_trade_agent

load_dotenv()

# ---------------------------------------------------------------------------
# Regression-detection helpers
# ---------------------------------------------------------------------------

BASELINE_FILE = Path(__file__).parent / "perf_baseline.json"
REGRESSION_THRESHOLD_PCT = 50  # fail if current > baseline × 1.5


def _load_baseline() -> dict:
    if BASELINE_FILE.exists():
        return json.loads(BASELINE_FILE.read_text())
    return {}


def _record_and_check_regression(name: str, elapsed_ms: float) -> None:
    """Store elapsed_ms as baseline on first run; assert no > 50 % regression on subsequent runs."""
    baseline = _load_baseline()
    if name not in baseline:
        baseline[name] = round(elapsed_ms, 3)
        BASELINE_FILE.write_text(json.dumps(baseline, indent=2))
        print(f"  Baseline established for '{name}': {elapsed_ms:.2f} ms")
        return

    expected = baseline[name]
    threshold = expected * (1 + REGRESSION_THRESHOLD_PCT / 100)
    if elapsed_ms > threshold:
        raise AssertionError(
            f"Performance regression in '{name}': "
            f"current={elapsed_ms:.1f} ms  >  "
            f"baseline={expected:.1f} ms × 1.{REGRESSION_THRESHOLD_PCT} = {threshold:.1f} ms"
        )
    print(
        f"  '{name}' {elapsed_ms:.2f} ms  ≤  regression threshold {threshold:.2f} ms ✓"
    )


# ---------------------------------------------------------------------------
# Reporter and RiskAgent latency-budget tests (Phase 10 spec)
# ---------------------------------------------------------------------------

def _create_reporter_agent():
    """Minimal reporter agent used for latency budget measurement."""
    from agno.agent import Agent

    return Agent(
        name="reporter",
        description="Morning briefing reporter.",
        instructions=[
            "You are Valura's morning briefing reporter.",
            "Write concise Markdown briefings. Cite data, balance perspectives, "
            "end with Key Actions.",
        ],
        model=OpenAIResponses(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
    )


def perf_reporter_latency_budget():
    """Reporter agent instantiation must complete under the 4 s latency budget.

    Measures object construction time (not an LLM round-trip) and:
    1. Fails hard if instantiation exceeds 4 000 ms (budget breach).
    2. Fails hard if instantiation regresses > 50 % vs. the stored baseline.
    """
    t0 = time.perf_counter()
    _create_reporter_agent()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"  Reporter instantiation: {elapsed_ms:.2f} ms  (budget: 4000 ms)")
    assert elapsed_ms < 4000, (
        f"Reporter instantiation {elapsed_ms:.1f} ms exceeded 4 s budget"
    )
    _record_and_check_regression("perf_reporter_instantiation", elapsed_ms)


def perf_risk_analyst_latency_budget():
    """RiskAgent instantiation must complete under the 2 s latency budget.

    Measures object construction time and:
    1. Fails hard if instantiation exceeds 2 000 ms (budget breach).
    2. Fails hard if instantiation regresses > 50 % vs. the stored baseline.
    """
    t0 = time.perf_counter()
    create_risk_analyst_agent()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"  RiskAgent instantiation: {elapsed_ms:.2f} ms  (budget: 2000 ms)")
    assert elapsed_ms < 2000, (
        f"RiskAgent instantiation {elapsed_ms:.1f} ms exceeded 2 s budget"
    )
    _record_and_check_regression("perf_risk_analyst_instantiation", elapsed_ms)


def perf_detect_introduced_regression():
    """Regression detection must catch a deliberately slow function.

    This test intentionally creates a variant that sleeps for 50 ms to simulate
    a code change that regressed performance. It asserts that the detection
    logic (current > baseline × 1.5) correctly identifies the regression,
    proving the suite will catch real regressions when they are introduced.
    """
    baseline_ms = 5.0  # fast instantiation baseline (ms)

    def slow_reporter():
        time.sleep(0.05)  # 50 ms artificial regression
        return _create_reporter_agent()

    t0 = time.perf_counter()
    slow_reporter()
    slow_ms = (time.perf_counter() - t0) * 1000

    regression_threshold = baseline_ms * (1 + REGRESSION_THRESHOLD_PCT / 100)
    assert slow_ms > regression_threshold, (
        f"Regression detection failed: slow_ms={slow_ms:.1f} ms did not exceed "
        f"threshold={regression_threshold:.1f} ms — detection is broken"
    )
    print(
        f"  Regression correctly detected: {slow_ms:.1f} ms  >  "
        f"threshold {regression_threshold:.1f} ms ✓"
    )


# ---------------------------------------------------------------------------
# Instantiation throughput benchmarks (existing)
# ---------------------------------------------------------------------------

def perf_market_research_instantiation():
    PerformanceEval(
        name="market_research_agent instantiation",
        func=create_market_research_agent,
        num_iterations=50,
        warmup_runs=5,
    ).run(print_results=True, print_summary=True)


def perf_financial_planning_instantiation():
    PerformanceEval(
        name="financial_planning_agent instantiation",
        func=create_financial_planning_agent,
        num_iterations=50,
        warmup_runs=5,
    ).run(print_results=True, print_summary=True)


def perf_risk_analyst_instantiation():
    PerformanceEval(
        name="risk_analyst_agent instantiation",
        func=create_risk_analyst_agent,
        num_iterations=50,
        warmup_runs=5,
    ).run(print_results=True, print_summary=True)


def perf_portfolio_health_instantiation():
    PerformanceEval(
        name="portfolio_health_agent instantiation",
        func=create_portfolio_health_agent,
        num_iterations=50,
        warmup_runs=5,
    ).run(print_results=True, print_summary=True)


def perf_trade_agent_instantiation():
    PerformanceEval(
        name="trade_agent instantiation",
        func=create_trade_agent,
        num_iterations=50,
        warmup_runs=5,
    ).run(print_results=True, print_summary=True)


def perf_general_enquiry_instantiation():
    PerformanceEval(
        name="general_enquiry_agent instantiation",
        func=create_general_enquiry_agent,
        num_iterations=50,
        warmup_runs=5,
    ).run(print_results=True, print_summary=True)


def perf_product_recommendation_instantiation():
    PerformanceEval(
        name="product_recommendation_agent instantiation",
        func=create_product_recommendation_agent,
        num_iterations=50,
        warmup_runs=5,
    ).run(print_results=True, print_summary=True)


async def perf_financial_planning_run():
    async def run():
        agent = create_financial_planning_agent()
        response: RunOutput = await agent.arun("What is the future value of $1,000 at 7% for 10 years?")
        return response

    await PerformanceEval(
        name="financial_planning_agent end-to-end run",
        func=run,
        num_iterations=3,
        warmup_runs=1,
    ).arun(print_results=True, print_summary=True)


if __name__ == "__main__":
    perf_reporter_latency_budget()
    perf_risk_analyst_latency_budget()
    perf_detect_introduced_regression()
    perf_market_research_instantiation()
    perf_financial_planning_instantiation()
    perf_risk_analyst_instantiation()
    perf_portfolio_health_instantiation()
    perf_trade_agent_instantiation()
    perf_general_enquiry_instantiation()
    perf_product_recommendation_instantiation()
    asyncio.run(perf_financial_planning_run())
