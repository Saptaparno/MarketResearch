import asyncio
import sys
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

from agents.backtest_agent import backtest_agent
from agents.market_research_agent import db


async def cancel_backtest(run_id: str, session_id: str | None = None) -> dict:
    accepted = await backtest_agent.acancel_run(run_id)

    await asyncio.sleep(4)

    result = await backtest_agent.aget_run_output(run_id, session_id=session_id)
    persisted_status = result.status.value if result else "not_found"

    return {
        "run_id": run_id,
        "cancel_accepted": accepted,
        "persisted_status": persisted_status,
    }


async def verify_schedule(name: str | None = None, runs_limit: int = 5) -> dict:
    schedules_page, total = await db.get_schedules(limit=100, page=1)

    if name:
        schedules_page = [s for s in schedules_page if s.get("name") == name]

    runs_by_schedule: dict = {}
    for sched in schedules_page:
        sid = sched.get("id") or sched.get("schedule_id")
        if sid:
            run_rows, _ = await db.get_schedule_runs(sid, limit=runs_limit, page=1)
            runs_by_schedule[sid] = run_rows

    return {"schedules": schedules_page, "runs": runs_by_schedule}


if __name__ == "__main__":
    import json

    async def demo():
        print("=== verify_schedule ===")
        data = await verify_schedule(name="morning_briefing")
        print(json.dumps(data, indent=2, default=str))

    asyncio.run(demo())
