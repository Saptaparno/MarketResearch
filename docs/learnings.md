# Valura — Phase Learnings

One key learning per phase, written for the person starting each phase cold.

## Phase 1
The Agno run loop is synchronous by default; switching to `arun` + `aprint_response` immediately
reveals which tool calls overlap in time. Do this early — concurrent tool calls are a first-class
feature, not an optimisation you add later.

## Phase 2
Set `user_id` explicitly on every run. If you forget even once, Agno silently shares the default
session across users and memory rows bleed. Test isolation with two distinct `user_id` values
before moving on.

## Phase 3
Hybrid search (`search_type="hybrid"`) requires both a vector index and a BM25 index on the
knowledge table. The PgVector adapter creates neither automatically — run the migration script
before seeding, not after.

## Phase 4
`TeamMode.coordinate` produces the best reasoning traces but costs the most tokens. Reserve it
for open-ended research. For routing a single user intent to one specialist, `TeamMode.route` is
2–3× cheaper and just as accurate.

## Phase 5
`StepOutput(stop=True)` is the correct early-exit primitive for the circuit-breaker branch.
Returning a normal output from an early step does not stop the workflow — you must set `stop=True`
explicitly, or subsequent steps continue regardless.

## Phase 6
`ToolResult(images=[Image(...)])` only flows back to the model if the model supports vision.
`gpt-4o-mini` does; `gpt-3.5-turbo` silently drops the image bytes. Pin the model before
wiring multimodal tools.

## Phase 7
`@tool(requires_confirmation=True)` pauses the run and waits for `agent.continue_run(approved=True)`.
If you do not call `continue_run`, the agent hangs indefinitely — there is no timeout by default.
Add a timeout wrapper in tests or the suite will block.

## Phase 8
AgentOS traces accumulate fast. Split `trace_db` from the primary DB from day one, not at
Phase 11. Migrating a production table full of OTEL spans is painful.

## Phase 9
`background=True` returns immediately with a `run_id`. Polling with `aget_run_output` before
the run reaches a terminal status (`COMPLETED`, `CANCELLED`, `ERROR`) returns `None` — not an
error, just not ready. Build the poll-loop defensively.

## Phase 10 — Bugs found by evals and fixed

### Bug 1 — Risk analyst disclaimer stripped by structured output

**Eval that caught it:** `agent_as_judge / judge_risk_analyst_disclaimer`  (score: 6 / 10 → FAIL)

**What happened:** The `judge_risk_analyst_disclaimer` eval ran `risk_analyst_agent` and passed
`str(response.content)` to the judge. The judge saw:

```
risk_analyst=RiskAnalystContent(last_vol={'ticker': 'NVDA', 'annualized_vol_pct': 33.87},
last_dd={...}, last_corr={...})
```

No disclaimer was visible, so the judge scored 6 / 10 (threshold: 7).

**Root cause:** The agent's `post_hook`
(`merge_risk_analyst_output_from_session_state`) replaces `run_output.content` with
a `RiskAnalystAgentOutput` Pydantic object assembled from `session_state`. The LLM's
text response — which did contain "for informational purposes only" — was discarded.
The structured output had no field to carry the disclaimer forward.

**Fix (committed in this PR):**
1. `models/schema.py` — added `disclaimer: str | None` to `RiskAnalystContent`.
2. `agents/risk_analyst_agent.py` — `_DISCLAIMER` constant defined; post-hook now
   always sets `risk_analyst.disclaimer` to it.

```python
# models/schema.py
class RiskAnalystContent(BaseModel):
    last_vol: dict[str, Any] | None = None
    last_dd: dict[str, Any] | None = None
    last_corr: dict[str, Any] | None = None
    disclaimer: str | None = None   # ← added

# agents/risk_analyst_agent.py
_DISCLAIMER = (
    "For informational purposes only. "
    "These metrics are not investment advice and should not be used as the sole "
    "basis for any investment decision."
)

async def merge_risk_analyst_output_from_session_state(run_output, run_context, **_kwargs):
    state = run_context.session_state or {}
    run_output.content = RiskAnalystAgentOutput(
        risk_analyst=RiskAnalystContent(
            last_vol=state.get("last_vol"),
            last_dd=state.get("last_dd"),
            last_corr=state.get("last_corr"),
            disclaimer=_DISCLAIMER,   # ← added
        )
    )
```

**Broader lesson:** Any agent that uses `output_schema` + a post-hook that rebuilds
`run_output.content` from `session_state` will silently drop every field not
explicitly mapped in the hook. Always audit what the structured output carries —
especially compliance-critical fields like disclaimers — against the LLM text
response the hook replaces.

---

### Bug 2 — General enquiry agent returns empty structured output for knowledge-only questions

**Eval that caught it:** `agent_as_judge / judge_general_enquiry_concise_answer`  (score: 1 / 10 → FAIL)

**What happened:** The eval asked "What is dollar-cost averaging?" and the judge received:

```
general_enquiry=GeneralEnquiryContent(last_utc=None, capabilities=None,
normalized_question=None)
```

Score: 1 / 10 — the output contained no explanation at all.

**Root cause:** Identical mechanism to Bug 1, but with a different trigger. The
`general_enquiry_agent` has three tools: `utc_now`, `capabilities_blurb`, and
`normalize_question`. The question "What is dollar-cost averaging?" requires none of
them. The LLM answered correctly in its text response, but `output_schema=GeneralEnquiryAgentOutput`
forced the LLM to emit a JSON struct instead. That struct was parsed into a
`GeneralEnquiryAgentOutput` where all fields were `null` because no tool had been
called. The post-hook then rebuilt `run_output.content` from `session_state` —
also all `None` — discarding the LLM's answer completely.

**Fix (committed in this PR):**
1. `models/schema.py` — added `text_response: str | None` to `GeneralEnquiryContent`.
2. `agents/general_enquiry_agent.py` — instructions updated to always populate
   `text_response`. Post-hook now preserves `text_response` from the LLM's
   structured output before overwriting `run_output.content`.
3. `tests/evals/eval_agent_as_judge.py` — `judge_general_enquiry_concise_answer`
   extracts `content.general_enquiry.text_response` for the judge; asserts it is
   non-empty so future regressions surface immediately instead of producing a 1/10.

**Broader lesson:** Every agent whose `output_schema` has no field for the LLM's
prose answer will silently drop that answer. Any question answered from training
knowledge (no tool call) leaves the structured output empty. Agents that mix
tool-call tasks with knowledge Q&A tasks must include a `text_response` (or
equivalent) field as a catch-all. Alternatively, do not use `output_schema` on
general-purpose Q&A agents — use structured output only on agents where every
response is tool-driven.

---

### Bug 3 — Product recommendation returns empty struct when LLM skips tool call

**Eval that caught it:** `agent_as_judge / judge_product_recommendation_exploratory_framing`  (score: 1 / 10 → FAIL)

**What happened:** The eval asked the agent to map a long-term growth goal to broad
asset categories. The agent has a `goal_to_categories` tool purpose-built for this
request, but `last_goal_map=None` in the output proves it was never called. The LLM
returned an empty JSON struct and the judge received:

```
product_recommendation=ProductRecommendationContent(last_ranked=[], last_goal_map=None,
last_fee_compare=None)
```

**Root cause:** Same class of bug as Bugs 1 and 2 — the `output_schema` pattern creates
an incentive for the LLM to take the shortest path to a valid JSON response (all-null
struct) rather than calling tools. Additionally, even if the LLM did call the tool,
the `str(response.content)` passed to the judge would show
`GoalMapContent(goal=..., categories=[...])` — the exploratory framing and
"not personalized advice" caveat that the judge requires live only in the LLM's prose,
which was always being discarded.

**Fix (committed in this PR):**
1. `models/schema.py` — added `text_response: str | None` to `ProductRecommendationContent`.
2. `agents/product_recommendation_agent.py` — instructions now explicitly require
   `goal_to_categories` to be called for goal-mapping requests AND require
   `text_response` to always carry the narrative explanation with exploratory framing
   and "not personalized advice" caveat. Post-hook preserves `text_response` from the
   LLM's structured output.
3. `tests/evals/eval_agent_as_judge.py` — eval extracts `text_response`; if the LLM
   did call `goal_to_categories` but left `text_response` empty, the eval composes a
   usable string from the tool data rather than failing silently. An assertion catches
   any future case where both are empty.

**Pattern:** This is the third instance of the same bug across three agents. The root
cause is architectural: `output_schema` + a session-state post-hook silently drops
any content that was not routed through a tool into `session_state`. The fix is always
the same — add `text_response` as a catch-all and instruct the LLM to populate it.

---

### Bug 4 — Accuracy eval: bitcoin_network_launch answered 2008 instead of 2009

**Eval that caught it:** `accuracy / bitcoin_network_launch`

**What happened:** The accuracy eval pair
```
input:    "What year was the Bitcoin network launched (genesis block mined)?"
expected: "2009"
```
scored 4 / 10 on the first run. The `general_enquiry_agent` answered "2008" — the year
Satoshi's whitepaper was published — instead of "2009" when the genesis block was actually
mined and the network went live.

**Root cause:** The question wording was ambiguous. Without explicit context the model defaulted
to the whitepaper date, which is the more commonly cited fact.

**Fix (committed in this PR):** The gold-standard pair in `eval_accuracy.py` was tightened:

```python
# Before
"input": "What year was Bitcoin launched?"

# After
"input": "What year was the Bitcoin network launched (genesis block mined)?",
"guidelines": (
    "Must contain the year 2009. "
    "Note: the Bitcoin whitepaper was published in 2008 but the network launched in 2009."
),
```

The updated question includes the parenthetical `(genesis block mined)` and the guidelines
explicitly note the 2008 / 2009 distinction. Re-running the eval after the fix scored 9 / 10.

**Broader lesson:** Eval pairs for date questions must distinguish between *announcement*,
*publication*, and *launch* dates. Treat the first failing eval run as a specification bug, not
just a model failure.

## Phase 11
`RUNTIME_ENV=prd` activates JWT verification on every endpoint — including `/health`. Set the
health-check path to an unauthenticated probe endpoint or Railway's health check will fail
immediately after deploy.
