from core.workflow import market_research_workflow
from agno.utils.pprint import pprint_run_response
import asyncio
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    user_input = "Get the historical data, stock info, and analyst recommendations for Apple (AAPL) over the last year."
    result = asyncio.run(market_research_workflow.arun(input=user_input))
    pprint_run_response(result)

    