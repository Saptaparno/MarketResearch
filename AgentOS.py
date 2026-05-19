from agents.backtest_agent import backtest_agent
from agents.multimodal_agent import multimodal_agent
from agents.market_research_agent import knowledge_base, db, market_research_agent
from agents.general_enquiry_agent import general_enquiry_agent
from agents.trade_agent import trade_agent
from agents.financial_planning_agent import financial_planning_agent
from agents.risk_analyst_agent import risk_analyst_agent
from agents.product_recommendation_agent import product_recommendation_agent
from agents.portfolio_health_agent import portfolio_health_agent
from core.workflow import morning_briefing_workflow, advisory_workflow, trade_workflow, multimodal_workflow, backtest_workflow, route_team, broadcast_team, coordinate_team
from agno.os import AgentOS
from dotenv import load_dotenv

import uvicorn

load_dotenv()


agent_os = AgentOS(
    id="agentos-demo",
    agents=[backtest_agent, multimodal_agent, market_research_agent, general_enquiry_agent, trade_agent, financial_planning_agent, risk_analyst_agent, product_recommendation_agent, portfolio_health_agent],
    teams=[route_team, broadcast_team, coordinate_team],
    workflows=[morning_briefing_workflow, advisory_workflow, trade_workflow, multimodal_workflow, backtest_workflow],
    db=db,
    knowledge=[knowledge_base],
    tracing=True,
    scheduler=True,
    scheduler_poll_interval=15,
    scheduler_base_url="http://127.0.0.1:8000",
)

app = agent_os.get_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
