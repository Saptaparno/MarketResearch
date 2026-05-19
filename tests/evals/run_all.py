import asyncio
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_accuracy import all_accuracy_evals
from eval_reliability import (
    eval_financial_planning_calls_fv_tool,
    eval_financial_planning_calls_monthly_tool,
    eval_financial_planning_calls_loan_tool,
    eval_portfolio_health_calls_snapshot_quote,
    eval_portfolio_health_calls_period_return,
    eval_risk_analyst_calls_volatility,
    eval_risk_analyst_calls_drawdown,
    eval_trade_agent_calls_propose_trade,
    eval_general_enquiry_calls_normalize,
    # Phase 10 morning-briefing workflow checks
    eval_morning_briefing_fetch_portfolio_once,
    eval_portfolio_drop_sets_red_flag,
    eval_risk_agent_triggered_on_drop,
)
from eval_performance import (
    perf_reporter_latency_budget,
    perf_risk_analyst_latency_budget,
    perf_detect_introduced_regression,
    perf_market_research_instantiation,
    perf_financial_planning_instantiation,
    perf_risk_analyst_instantiation,
    perf_portfolio_health_instantiation,
    perf_trade_agent_instantiation,
    perf_general_enquiry_instantiation,
    perf_product_recommendation_instantiation,
    perf_financial_planning_run,
)
from eval_agent_as_judge import (
    # Phase 10 reporter judge checks
    judge_reporter_evidence_cited,
    judge_reporter_balanced,
    judge_reporter_actionable,
    # Original judge checks
    judge_market_research_structure,
    judge_risk_analyst_disclaimer,
    judge_financial_planning_educational_tone,
    judge_general_enquiry_concise_answer,
    judge_product_recommendation_exploratory_framing,
)

SUITES = {
    "accuracy": all_accuracy_evals(),  # 20 gold-standard pairs
    "reliability": [
        eval_financial_planning_calls_fv_tool,
        eval_financial_planning_calls_monthly_tool,
        eval_financial_planning_calls_loan_tool,
        eval_portfolio_health_calls_snapshot_quote,
        eval_portfolio_health_calls_period_return,
        eval_risk_analyst_calls_volatility,
        eval_risk_analyst_calls_drawdown,
        eval_trade_agent_calls_propose_trade,
        eval_general_enquiry_calls_normalize,
        eval_morning_briefing_fetch_portfolio_once,
        eval_portfolio_drop_sets_red_flag,
        eval_risk_agent_triggered_on_drop,
    ],
    "performance": [
        perf_reporter_latency_budget,
        perf_risk_analyst_latency_budget,
        perf_detect_introduced_regression,
        perf_market_research_instantiation,
        perf_financial_planning_instantiation,
        perf_risk_analyst_instantiation,
        perf_portfolio_health_instantiation,
        perf_trade_agent_instantiation,
        perf_general_enquiry_instantiation,
        perf_product_recommendation_instantiation,
        perf_financial_planning_run,
    ],
    "agent_as_judge": [
        judge_reporter_evidence_cited,
        judge_reporter_balanced,
        judge_reporter_actionable,
        judge_market_research_structure,
        judge_risk_analyst_disclaimer,
        judge_financial_planning_educational_tone,
        judge_general_enquiry_concise_answer,
        judge_product_recommendation_exploratory_framing,
    ],
}


async def _run_async(suite: str | None = None):
    targets = {suite: SUITES[suite]} if suite else SUITES
    passed, failed = 0, []

    for name, fns in targets.items():
        print(f"\n{'='*60}")
        print(f"  {name.upper()} EVALS")
        print(f"{'='*60}")
        for fn in fns:
            try:
                if inspect.iscoroutinefunction(fn):
                    await fn()
                else:
                    fn()
                passed += 1
                print(f"  PASSED {fn.__name__}")
            except Exception as exc:
                failed.append((fn.__name__, exc))
                print(f"  FAILED {fn.__name__}: {exc}")

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {len(failed)} failed")
    if failed:
        print("  Failed tests:")
        for name, exc in failed:
            print(f"    - {name}: {exc}")
    print(f"{'='*60}")
    return failed


def run(suite: str | None = None):
    failed = asyncio.run(_run_async(suite))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    suite_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if suite_arg and suite_arg not in SUITES:
        print(f"Unknown suite '{suite_arg}'. Choose from: {', '.join(SUITES)}")
        sys.exit(1)
    run(suite_arg)
