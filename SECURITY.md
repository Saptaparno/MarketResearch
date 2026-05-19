# SECURITY.md — Valura Trade Safety

## Overview

All trade proposals pass through three independent safety tiers before any
order is recorded. A failure at any tier aborts the action and returns a
structured error to the caller.

---

## Tier 1 — User Confirmation (in-process)

| Property | Detail |
|---|---|
| Agno primitive | `@tool(requires_confirmation=True)` |
| Tools covered | `propose_trade` |
| How it works | The agent run **pauses** before executing the tool and returns `RunEvent.tool_approval_requested`. The caller must call `agent.continue_run(requirements=[...], confirmation=True/False)` to resume or cancel. |
| On rejection | The tool is **never executed**. The run completes with a cancellation message. |
| DB record | None — entirely in-process. |

---

## Tier 2 — Admin Approval (database-backed)

| Property | Detail |
|---|---|
| Agno primitive | `@approval(type="required")` + `@tool(requires_confirmation=True)` |
| Tools covered | `propose_large_trade` (notional > $10K) |
| How it works | Same pause as Tier 1, **plus** a row is written to the `agno_approvals` table with `approval_type = 'required'` and `status = 'pending'`. An out-of-band admin approver loop (see `fake_admin_approver` in `agents/trade_agent.py`) queries the table and calls `agent.continue_run` with the admin's decision. |
| On rejection | Tool not executed; `agno_approvals` row updated to `status = 'rejected'`. |
| Audit variant | `audit_large_trade` uses `@approval(type="audit")`, which writes to `agno_approvals` **after** the HITL resolves (non-blocking audit trail). |
| DB record | `agno_approvals` table (managed by Agno). |

---

## Tier 3 — Input Guardrails + Background Audit

### Pre-hooks (checked before the model sees any input)

| Guardrail | Class | Behaviour on trigger |
|---|---|---|
| PII masking | `PIIDetectionGuardrail(mask_pii=True)` | SSNs, credit-card numbers, emails, and phone numbers are **masked with `*`** before the prompt reaches the model. |
| Prompt injection | `PromptInjectionGuardrail()` | Raises `InputCheckError` (fails closed) if the input matches known injection patterns (e.g. "ignore previous instructions", "jailbreak"). |
| Ticker allowlist | `TickerPermissionGuardrail(permitted_tickers=[...])` | Raises `InputCheckError` if the input mentions a ticker not in the user's allowlist. Pass `None` to disable (unrestricted). |

Guardrails are passed as `pre_hooks` on the `trade_agent`. They run
synchronously before every LLM call and cannot be bypassed by the model.

### Post-hook (background audit)

| Hook | Attribute | Behaviour |
|---|---|---|
| `audit_log_hook` | `@hook(run_in_background=True)` | After the agent responds, writes one row to the `audit_log` table (defined in `db/audit_log.py`). Runs in the background so it **never adds latency** to the user-facing response. |

The `audit_log` table columns: `id`, `ts`, `user_id`, `run_id`, `action`,
`ticker`, `side`, `qty`, `notional`, `approved`, `meta` (JSONB).

---

## Summary

```
User input
    │
    ▼
[Pre-hooks] PIIDetectionGuardrail → PromptInjectionGuardrail → TickerPermissionGuardrail
    │ pass
    ▼
Model decides to call propose_trade / propose_large_trade
    │
    ▼
[Tier 1/2] Run pauses → caller approves or rejects
    │ approved
    ▼
Tool executes → result returned to model
    │
    ▼
[Post-hook background] audit_log_hook writes to audit_log table
```

Any rejection at any stage leaves no executed trade. Every approved action
has a row in `audit_log`. Large trades additionally have a row in
`agno_approvals`.
