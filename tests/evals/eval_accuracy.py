import asyncio
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agno.eval.accuracy import AccuracyEval, AccuracyResult
from agno.models.openai import OpenAIResponses
from dotenv import load_dotenv

from agents.general_enquiry_agent import create_general_enquiry_agent

load_dotenv()

EVALUATOR_MODEL = OpenAIResponses(id="gpt-4o-mini")

# 20 gold-standard pairs: historical events and facts with public, verifiable outcomes
_GOLD_PAIRS = [
    {
        "name": "apple_founding_year",
        "input": "What year was Apple Inc. founded?",
        "expected": "1976",
        "guidelines": "Must contain the year 1976.",
    },
    {
        "name": "apple_ceo",
        "input": "Who is the current CEO of Apple Inc.?",
        "expected": "Tim Cook",
        "guidelines": "Must mention Tim Cook.",
    },
    {
        "name": "amazon_founding_year",
        "input": "What year was Amazon founded?",
        "expected": "1994",
        "guidelines": "Must contain the year 1994.",
    },
    {
        "name": "amazon_founder",
        "input": "Who founded Amazon?",
        "expected": "Jeff Bezos",
        "guidelines": "Must mention Jeff Bezos.",
    },
    {
        "name": "microsoft_founding_year",
        "input": "What year was Microsoft founded?",
        "expected": "1975",
        "guidelines": "Must contain the year 1975.",
    },
    {
        "name": "google_ipo_year",
        "input": "What year did Google (Alphabet) complete its initial public offering?",
        "expected": "2004",
        "guidelines": "Must contain the year 2004.",
    },
    {
        "name": "facebook_ipo_year",
        "input": "What year did Facebook complete its IPO on NASDAQ?",
        "expected": "2012",
        "guidelines": "Must contain the year 2012.",
    },
    {
        "name": "nvidia_ticker_company",
        "input": "What company does the stock ticker symbol NVDA represent?",
        "expected": "NVIDIA Corporation",
        "guidelines": "Must mention NVIDIA.",
    },
    {
        "name": "berkshire_ceo",
        "input": "Who is the CEO of Berkshire Hathaway?",
        "expected": "Warren Buffett",
        "guidelines": "Must mention Warren Buffett.",
    },
    {
        "name": "financial_crisis_2008_cause",
        "input": "What financial instruments were at the centre of the 2008 financial crisis?",
        "expected": "mortgage-backed securities or subprime mortgages",
        "guidelines": "Must mention subprime mortgages, mortgage-backed securities, or the housing bubble.",
    },
    {
        "name": "sp500_definition",
        "input": "What is the S&P 500 index?",
        "expected": "A stock market index tracking 500 large US publicly traded companies",
        "guidelines": "Must describe it as a stock market index of approximately 500 large US companies.",
    },
    {
        "name": "bear_market_definition",
        "input": "What is the formal definition of a bear market?",
        "expected": "A decline of 20% or more from recent highs",
        "guidelines": "Must mention a 20% decline from recent highs.",
    },
    {
        "name": "etf_expansion",
        "input": "What does the acronym ETF stand for in finance?",
        "expected": "Exchange-Traded Fund",
        "guidelines": "Must say Exchange-Traded Fund.",
    },
    {
        "name": "ipo_expansion",
        "input": "What does IPO stand for?",
        "expected": "Initial Public Offering",
        "guidelines": "Must say Initial Public Offering.",
    },
    {
        "name": "pe_ratio_definition",
        "input": "What does the P/E ratio measure in stock analysis?",
        "expected": "It compares a stock's price to its earnings per share",
        "guidelines": "Must describe comparing stock price to earnings per share.",
    },
    {
        "name": "federal_reserve_mandate",
        "input": "What are the two goals of the US Federal Reserve's dual mandate?",
        "expected": "Maximum employment and price stability",
        "guidelines": "Must mention both employment and price stability or inflation control.",
    },
    {
        "name": "tesla_founding_year",
        "input": "What year was Tesla Inc. incorporated?",
        "expected": "2003",
        "guidelines": "Must contain the year 2003.",
    },
    {
        "name": "bitcoin_network_launch",
        "input": "What year was the Bitcoin network launched when the genesis block was mined?",
        "expected": "2009",
        "guidelines": (
            "Must contain the year 2009. "
            "Note: the Bitcoin whitepaper was published in 2008 but the network launched in 2009."
        ),
    },
    {
        "name": "black_monday_year",
        "input": "In what year did the stock market crash known as Black Monday occur?",
        "expected": "1987",
        "guidelines": "Must contain the year 1987.",
    },
    {
        "name": "dollar_cost_averaging",
        "input": "What is dollar-cost averaging as an investment strategy?",
        "expected": "Investing a fixed amount at regular intervals regardless of price",
        "guidelines": "Must mention regular or fixed-interval investing regardless of market price.",
    },
]


def _make_accuracy_eval(pair: dict):
    async def fn():
        result: Optional[AccuracyResult] = await AccuracyEval(
            name=pair["name"],
            model=EVALUATOR_MODEL,
            agent=create_general_enquiry_agent(structured_output=False),
            input=pair["input"],
            expected_output=pair["expected"],
            additional_guidelines=pair["guidelines"],
            num_iterations=1,
        ).arun(print_results=True)
        assert result is not None and result.avg_score >= 7, (
            f"Accuracy eval '{pair['name']}' scored "
            f"{result.avg_score if result else 'N/A'} / 10 (threshold: 7)"
        )

    fn.__name__ = f"eval_{pair['name']}"
    return fn


# Generate module-level functions for all 20 pairs so they are individually importable
for _pair in _GOLD_PAIRS:
    globals()[f"eval_{_pair['name']}"] = _make_accuracy_eval(_pair)


def all_accuracy_evals() -> list:
    return [globals()[f"eval_{p['name']}"] for p in _GOLD_PAIRS]


if __name__ == "__main__":
    for _fn in all_accuracy_evals():
        asyncio.run(_fn())
