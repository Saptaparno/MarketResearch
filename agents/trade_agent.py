"""Trade agent — Phase 7: HITL, guardrails, hooks.

Three HITL tiers
----------------
Tier 1  propose_trade           @tool(requires_confirmation=True)
        Pauses the run; caller calls continue_run(confirmed=True/False).
        No DB record.

Tier 2  propose_large_trade     @approval(type="required") + requires_confirmation
        Same pause, but also writes a row to agno_approvals so an admin
        approver loop can inspect and resolve it out-of-band.

Tier 2b audit_large_trade       @approval(type="audit") + requires_confirmation
        Writes an agno_approvals audit row AFTER the HITL resolves.

Pre-hooks (guardrails, checked before the model sees the input)
---------------------------------------------------------------
  PIIDetectionGuardrail(mask_pii=True)  — masks SSN / CC / email / phone
  PromptInjectionGuardrail()            — rejects known injection payloads
  TickerPermissionGuardrail(...)        — blocks forbidden tickers (per user)

Post-hook (non-blocking)
------------------------
  audit_log_hook  @hook(run_in_background=True)
  Writes one row to the audit_log table after every run so every approved
  action is persisted independently of the agno_approvals table.
"""

import os
import sys
from pathlib import Path

from agno.agent import Agent
from agno.approval.decorator import approval
from agno.guardrails.pii import PIIDetectionGuardrail
from agno.guardrails.prompt_injection import PromptInjectionGuardrail
from agno.hooks.decorator import hook
from agno.models.openai import OpenAIResponses
from agno.run import RunContext
from agno.tools import tool
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from agents.market_research_agent import db, knowledge_base
from sqlalchemy.sql.expression import text as sql_text
from guardrails.ticker_guardrail import TickerPermissionGuardrail
from models.schema import TradeAgentOutput, TradeContent

load_dotenv()

# ---------------------------------------------------------------------------
# Tier 1 — user-facing confirmation (no DB record)
# ---------------------------------------------------------------------------

@tool(requires_confirmation=True)
async def propose_trade(run_context: RunContext, ticker: str, side: str, qty: float):
    """Propose a trade that requires user confirmation before it is recorded.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL).
        side:   'buy' or 'sell'.
        qty:    Number of shares.
    """
    ticker = ticker.upper().strip()
    side = side.lower().strip()
    if side not in ("buy", "sell"):
        err = {"error": f"Invalid side {side!r}. Must be 'buy' or 'sell'."}
        run_context.session_state["last_propose"] = err
        return TradeContent(last_propose=err)

    record = {"ticker": ticker, "side": side, "qty": qty, "status": "proposed"}
    run_context.session_state["last_propose"] = record
    return TradeContent(last_propose=record)


# ---------------------------------------------------------------------------
# Tier 2 — admin approval required (writes to agno_approvals)
# ---------------------------------------------------------------------------

@approval(type="required")
@tool(requires_confirmation=True)
async def propose_large_trade(run_context: RunContext, ticker: str, side: str, qty: float, notional: float):
    """Propose a large trade (notional > $10K). Requires admin approval.

    Args:
        ticker:   Stock ticker symbol.
        side:     'buy' or 'sell'.
        qty:      Number of shares.
        notional: Total dollar value of the trade.
    """
    ticker = ticker.upper().strip()
    side = side.lower().strip()

    if notional <= 10_000:
        note = {"warning": "notional <= $10K — use propose_trade instead", "notional": notional}
        run_context.session_state["last_large_propose"] = note
        return TradeContent(last_large_propose=note)

    record = {
        "ticker": ticker,
        "side": side,
        "qty": qty,
        "notional": notional,
        "status": "pending_admin_approval",
    }
    run_context.session_state["last_large_propose"] = record
    return TradeContent(last_large_propose=record)


# ---------------------------------------------------------------------------
# Tier 2b — audit trail (writes agno_approvals row after HITL resolves)
# ---------------------------------------------------------------------------

@approval(type="audit")
@tool(requires_confirmation=True)
async def audit_large_trade(run_context: RunContext, ticker: str, side: str, qty: float, notional: float):
    """Record an audit entry for a large trade that was confirmed by the user.

    Args:
        ticker:   Stock ticker symbol.
        side:     'buy' or 'sell'.
        qty:      Number of shares.
        notional: Total dollar value of the trade.
    """
    ticker = ticker.upper().strip()
    record = {
        "ticker": ticker,
        "side": side.lower().strip(),
        "qty": qty,
        "notional": notional,
        "status": "audit_recorded",
    }
    run_context.session_state["last_large_propose"] = record
    return TradeContent(last_large_propose=record)


# ---------------------------------------------------------------------------
# Post-hook — background audit_log write (never blocks the response)
# ---------------------------------------------------------------------------

async def _write_audit(run_output, run_context, user_id):
    import json
    from datetime import datetime, timezone

    state = run_context.session_state if run_context else {}
    propose = state.get("last_propose") or state.get("last_large_propose")
    if not propose:
        return
    try:
        async with db._engine.begin() as conn:
            await conn.execute(
                sql_text("""
                    INSERT INTO audit_log
                        (ts, user_id, run_id, action, ticker, side, qty, notional, approved, meta)
                    VALUES
                        (:ts, :user_id, :run_id, :action, :ticker, :side, :qty, :notional, :approved, :meta)
                """),
                {
                    "ts": datetime.now(timezone.utc),
                    "user_id": user_id,
                    "run_id": getattr(run_output, "run_id", None),
                    "action": "propose_large_trade" if "notional" in propose else "propose_trade",
                    "ticker": propose.get("ticker"),
                    "side": propose.get("side"),
                    "qty": propose.get("qty"),
                    "notional": propose.get("notional"),
                    "approved": propose.get("status") not in ("error",),
                    "meta": json.dumps(propose),
                },
            )
    except Exception:
        pass


@hook(run_in_background=True)
async def audit_log_hook(run_output, run_context, user_id=None, **_kwargs):
    import asyncio
    asyncio.create_task(_write_audit(run_output, run_context, user_id))


# ---------------------------------------------------------------------------
# Output merger post-hook
# ---------------------------------------------------------------------------

async def merge_trade_output(run_output, run_context, **_kwargs):
    state = run_context.session_state or {}
    run_output.content = TradeAgentOutput(
        trade=TradeContent(
            last_propose=state.get("last_propose"),
            last_large_propose=state.get("last_large_propose"),
        )
    )


# ---------------------------------------------------------------------------
# Admin approver loop (fake — simulates an out-of-band admin)
# ---------------------------------------------------------------------------

async def fake_admin_approver(agent: Agent, run_response, *, approved: bool = True) -> None:
    """Simulate an admin resolving a pending agno_approvals row.

    In production this would be a webhook handler or a background service that
    queries agno_approvals for status='pending' and calls continue_run after a
    human clicks Approve/Reject in a UI.

    Args:
        agent:        The trade_agent instance whose run is paused.
        run_response: The paused RunOutput returned by agent.arun().
        approved:     Whether the admin approves (True) or rejects (False).
    """

    if not run_response or not run_response.requirements:
        return

    updated_requirements = []
    for req in run_response.requirements:
        req.confirmation = approved
        req.confirmation_note = "admin-approved" if approved else "admin-rejected"
        if req.tool_execution is not None:
            req.tool_execution.confirmed = approved
        updated_requirements.append(req)

    await agent.acontinue_run(
        run_response=run_response,
        requirements=updated_requirements,
    )


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def trade_tools_factory(agent=None, run_context=None):
    return [propose_trade, propose_large_trade, audit_large_trade]


def create_trade_agent(permitted_tickers: list[str] | None = None) -> Agent:
    """Create the trade agent.

    Args:
        permitted_tickers: Tickers this user is allowed to trade.
                           Pass None to disable the allowlist guardrail.
    """
    guardrails = [
        PIIDetectionGuardrail(mask_pii=True),
        PromptInjectionGuardrail(),
        TickerPermissionGuardrail(permitted_tickers=permitted_tickers),
    ]

    return Agent(
        db=db,
        knowledge=knowledge_base,
        search_knowledge=False,
        add_knowledge_to_context=False,
        name="trade_agent",
        description=(
            "Proposes trades for human review. All trades require explicit user confirmation "
            "(Tier 1). Large trades (notional > $10K) additionally route to an admin approver "
            "(Tier 2). Every action is written to audit_log (Tier 3)."
        ),
        instructions=(
            "You help users propose trades. Use propose_trade for standard orders and "
            "propose_large_trade when the notional value exceeds $10,000. Never execute "
            "a trade without explicit confirmation. Do not give investment advice."
        ),
        reasoning=True,
        tools=trade_tools_factory,
        pre_hooks=guardrails,
        post_hooks=[merge_trade_output, audit_log_hook],
        add_history_to_context=False,
        add_session_state_to_context=True,
        session_state={"last_propose": None, "last_large_propose": None},
        output_schema=TradeAgentOutput,
        model=OpenAIResponses(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
    )


trade_agent = create_trade_agent()
