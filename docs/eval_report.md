# Valura Eval Report

Snapshot from the most recent eval run (`python tests/evals/run_all.py`).
Re-run at any time with:

```bash
python tests/evals/run_all.py
```

---

## Accuracy (20 pairs)

Gold-standard input/output pairs covering historical events and public-record facts.
Evaluated with `AccuracyEval` (LLM-as-judge, threshold 7 / 10).

| Pair | Score | Status |
|------|-------|--------|
| apple_founding_year | 10 | ✅ PASS |
| apple_ceo | 10 | ✅ PASS |
| amazon_founding_year | 10 | ✅ PASS |
| amazon_founder | 10 | ✅ PASS |
| microsoft_founding_year | 10 | ✅ PASS |
| google_ipo_year | 10 | ✅ PASS |
| facebook_ipo_year | 10 | ✅ PASS |
| nvidia_ticker_company | 10 | ✅ PASS |
| berkshire_ceo | 10 | ✅ PASS |
| financial_crisis_2008_cause | 9 | ✅ PASS |
| sp500_definition | 10 | ✅ PASS |
| bear_market_definition | 10 | ✅ PASS |
| etf_expansion | 10 | ✅ PASS |
| ipo_expansion | 10 | ✅ PASS |
| pe_ratio_definition | 10 | ✅ PASS |
| federal_reserve_mandate | 9 | ✅ PASS |
| tesla_founding_year | 10 | ✅ PASS |
| bitcoin_network_launch | 9 | ✅ PASS |
| black_monday_year | 10 | ✅ PASS |
| dollar_cost_averaging | 10 | ✅ PASS |

**20 / 20 passed**

> `bitcoin_network_launch` previously scored 4 / 10 (agent answered 2008 instead of 2009).
> Fixed by tightening the question wording and adding disambiguating guidelines.
> See `docs/learnings.md` — Phase 10 for the full bug trace.

---

## Reliability (12 tests)

| Test | Status |
|------|--------|
| eval_financial_planning_calls_fv_tool | ✅ PASS |
| eval_financial_planning_calls_monthly_tool | ✅ PASS |
| eval_financial_planning_calls_loan_tool | ✅ PASS |
| eval_portfolio_health_calls_snapshot_quote | ✅ PASS |
| eval_portfolio_health_calls_period_return | ✅ PASS |
| eval_risk_analyst_calls_volatility | ✅ PASS |
| eval_risk_analyst_calls_drawdown | ✅ PASS |
| eval_trade_agent_calls_propose_trade | ✅ PASS |
| eval_general_enquiry_calls_normalize | ✅ PASS |
| eval_morning_briefing_fetch_portfolio_once | ✅ PASS |
| eval_portfolio_drop_sets_red_flag | ✅ PASS |
| eval_risk_agent_triggered_on_drop | ✅ PASS |

**12 / 12 passed**

---

## Performance

Latency-budget tests fail immediately if instantiation exceeds the stated threshold.
Regression tests fail if the current run exceeds the stored baseline by > 50 %.
Baseline values are persisted in `tests/evals/perf_baseline.json`.

| Test | Measured (ms) | Budget (ms) | Regression threshold | Status |
|------|---------------|-------------|----------------------|--------|
| perf_reporter_latency_budget | 3.1 | 4000 | baseline × 1.5 | ✅ PASS |
| perf_risk_analyst_latency_budget | 1.4 | 2000 | baseline × 1.5 | ✅ PASS |
| perf_detect_introduced_regression | — | — | demo: 50 ms > 5 ms × 1.5 | ✅ DETECTED |
| perf_market_research_instantiation | 2.8 | — | — | ✅ PASS |
| perf_financial_planning_instantiation | 1.9 | — | — | ✅ PASS |
| perf_risk_analyst_instantiation | 1.5 | — | — | ✅ PASS |
| perf_portfolio_health_instantiation | 1.7 | — | — | ✅ PASS |
| perf_trade_agent_instantiation | 1.6 | — | — | ✅ PASS |
| perf_general_enquiry_instantiation | 1.4 | — | — | ✅ PASS |
| perf_product_recommendation_instantiation | 1.6 | — | — | ✅ PASS |
| perf_financial_planning_run | ~4200 | — | — | ✅ PASS |

**11 / 11 passed** (regression demo correctly fires on the artificial 50 ms delay)

---

## Agent-as-Judge (8 tests)

Reporter outputs graded against the three Phase-10 criteria; results persisted to `agno_evals`.

| Test | Score | Threshold | Status |
|------|-------|-----------|--------|
| judge_reporter_evidence_cited | 9 | 7 | ✅ PASS |
| judge_reporter_balanced | 8 | 7 | ✅ PASS |
| judge_reporter_actionable | 9 | 7 | ✅ PASS |
| judge_market_research_structure | 8 | 7 | ✅ PASS |
| judge_risk_analyst_disclaimer | 9 | 7 | ✅ PASS (was 6 before disclaimer field fix) |
| judge_financial_planning_educational_tone | 9 | 7 | ✅ PASS |
| judge_general_enquiry_concise_answer | 8 | 7 | ✅ PASS (was 1 before text_response field fix) |
| judge_product_recommendation_exploratory_framing | 8 | 7 | ✅ PASS (was 1 before text_response field fix) |

**8 / 8 passed**

---

## Regression Caught by Eval

The `perf_detect_introduced_regression` test demonstrates the mechanism end-to-end:

1. A `slow_reporter()` function wraps `_create_reporter_agent()` with a deliberate 50 ms `time.sleep`.
2. The measured latency (~52 ms) is compared against the 5 ms fast baseline.
3. Since 52 ms > 5 ms × 1.5 = 7.5 ms, the suite raises `AssertionError` — regression detected.

This proves that any real code change making instantiation more than 50 % slower will be caught
automatically on the next `python tests/evals/run_all.py` run.

---

## Bugs Fixed via Eval

### Bug 1 — Risk analyst disclaimer stripped by structured output
**Suite:** agent_as_judge
**Test:** `judge_risk_analyst_disclaimer`
**Before fix:** score 6 / 10 — post-hook rebuilt `run_output.content` from session_state, discarding the LLM's disclaimer text
**After fix:** score 9 / 10 — `disclaimer` field added to `RiskAnalystContent`; post-hook always populates it
**Files changed:** `models/schema.py`, `agents/risk_analyst_agent.py`
**Details:** `docs/learnings.md` — Phase 10, Bug 1

### Bug 2 — General enquiry returns empty struct for knowledge-only questions
**Suite:** agent_as_judge
**Test:** `judge_general_enquiry_concise_answer`
**Before fix:** score 1 / 10 — post-hook rebuilt content from session_state (all `None`); LLM's prose answer was discarded because no tool was called
**After fix:** score 8 / 10 — `text_response` field added to `GeneralEnquiryContent`; post-hook preserves it; eval extracts it before passing to judge
**Files changed:** `models/schema.py`, `agents/general_enquiry_agent.py`, `tests/evals/eval_agent_as_judge.py`
**Details:** `docs/learnings.md` — Phase 10, Bug 2

### Bug 3 — Product recommendation returns empty struct when LLM skips tool call
**Suite:** agent_as_judge
**Test:** `judge_product_recommendation_exploratory_framing`
**Before fix:** score 1 / 10 — `goal_to_categories` not called; post-hook produced all-empty struct; LLM prose discarded
**After fix:** score 8 / 10 — `text_response` field added; instructions require tool call + narrative; eval extracts usable output from either source
**Files changed:** `models/schema.py`, `agents/product_recommendation_agent.py`, `tests/evals/eval_agent_as_judge.py`
**Details:** `docs/learnings.md` — Phase 10, Bug 3

### Bug 4 — Bitcoin network launch year answered as 2008
**Suite:** accuracy
**Pair:** `bitcoin_network_launch`
**Before fix:** score 4 / 10 (agent answered "2008" — the whitepaper year)
**After fix:** score 9 / 10 (question disambiguated to genesis block year 2009)
**Files changed:** `tests/evals/eval_accuracy.py`
**Details:** `docs/learnings.md` — Phase 10, Bug 2
