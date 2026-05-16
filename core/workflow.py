from agno.workflow import Workflow
from agno.workflow.step import Step

from agents.market_research_agent import market_research_agent

market_research_workflow = Workflow(
    name="market_research_workflow",
    steps=[Step(agent=market_research_agent)],
)
