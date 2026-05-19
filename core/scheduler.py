import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agno.scheduler import ScheduleManager
from agents.market_research_agent import db
from dotenv import load_dotenv

load_dotenv()

schedule_manager = ScheduleManager(db=db)
async def schedule_morning_briefing():
    schedule = await schedule_manager.acreate(
    name="morning_briefing",
    cron="0 6 * * *",
    endpoint="/workflows/morning_briefing/runs",
    description="Daily morning market briefing at 6 AM UTC",
    payload={
        "message": "Give me today's morning market briefing with portfolio health and macro outlook.",
    },
    timezone="Asia/Kolkata",
    if_exists="skip",
)

    return schedule


if __name__ == "__main__":
    import asyncio
    schedule = asyncio.run(schedule_morning_briefing())
    print(f"Scheduled: {schedule}")