import asyncio
import os
import re

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.team import Team, TeamMode
from agno.tools.discord import DiscordTools
from agno.workflow import Condition, Loop, Parallel, Router, Step, StepInput, StepOutput, Workflow

from agents.backtest_agent import backtest_agent
from agents.financial_planning_agent import financial_planning_agent
from agents.general_enquiry_agent import general_enquiry_agent
from agents.market_research_agent import db, market_research_agent
from agents.multimodal_agent import multimodal_agent
from agents.portfolio_health_agent import portfolio_health_agent
from agents.product_recommendation_agent import product_recommendation_agent
from agents.risk_analyst_agent import risk_analyst_agent
from agents.trade_agent import trade_agent


def team_leader_model():
    return OpenAIResponses(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))


ADVISORY_MEMBERS = [
    portfolio_health_agent,
    risk_analyst_agent,
    market_research_agent,
    financial_planning_agent,
    product_recommendation_agent,
    general_enquiry_agent,
    trade_agent,
]

coordinate_team = Team(
    name="coordinate_team",
    mode=TeamMode.coordinate,
    members=ADVISORY_MEMBERS,
    model=team_leader_model(),
    db=db,
    markdown=True,
    search_knowledge=True,
    instructions=(
        "Coordinate mode: open-ended tasks. Break down the request, delegate to the right specialists, "
        "iterate if needed, then produce one clear combined answer."
    ),
)

route_team = Team(
    name="route_team",
    mode=TeamMode.route,
    members=ADVISORY_MEMBERS,
    model=team_leader_model(),
    db=db,
    markdown=True,
    search_knowledge=True,
    instructions=(
        "Route mode: infer the user's intent, choose exactly one member agent who is the best fit, "
        "delegate once, and return that member's response as the team output."
    ),
)

broadcast_team = Team(
    name="broadcast_team",
    mode=TeamMode.broadcast,
    members=ADVISORY_MEMBERS,
    model=team_leader_model(),
    db=db,
    markdown=True,
    search_knowledge=True,
    instructions=(
        "Broadcast mode: when multiple perspectives help, send the same task to all members, "
        "then merge their inputs into a single coherent reply."
    ),
)

async def coordinate_step(step_input: StepInput, session_state: dict) -> StepOutput:
    resp = await coordinate_team.arun(step_input.input)
    content = to_str(resp)
    session_state["coordinate_output"] = content
    return StepOutput(step_name="coordinate_step", content=content, success=True)


async def route_step(step_input: StepInput, session_state: dict) -> StepOutput:
    prev = step_input.get_step_content("coordinate_step") or step_input.input
    resp = await route_team.arun(prev)
    content = to_str(resp)
    session_state["route_output"] = content
    return StepOutput(step_name="route_step", content=content, success=True)


async def broadcast_step(step_input: StepInput, session_state: dict) -> StepOutput:
    prev = step_input.get_step_content("route_step") or step_input.input
    resp = await broadcast_team.arun(prev)
    content = to_str(resp)
    session_state["broadcast_output"] = content
    return StepOutput(step_name="broadcast_step", content=content, success=True)


advisory_workflow = Workflow(
    id="advisory_workflow",
    name="advisory_workflow",
    db=db,
    session_state={},
    steps=[
        coordinate_step,
        route_step,
        broadcast_step,
    ],
)


def to_str(response):
    if response is None:
        return ""
    c = response.content
    return c if isinstance(c, str) else str(c)


async def fetch_portfolio(_, session_state):
    portfolio = session_state.get("portfolio_tickers", ["NVDA", "AAPL", "BND"])
    session_state.update(research_idx=0, ticker_research={})

    all_returns = {}
    for ticker in portfolio:
        resp = await portfolio_health_agent.arun(f"Get the 1-year total return percentage for {ticker}.")
        ret_val = 0.0
        c = resp.content if resp else None
        if c is not None and hasattr(c, "portfolio_health"):
            lr = c.portfolio_health.last_return
            if isinstance(lr, dict):
                ret_val = float(lr.get("total_return_pct") or 0.0)
        all_returns[ticker] = ret_val

    flagged = [t for t, v in all_returns.items() if v < -5.0]
    session_state.update(portfolio_returns=all_returns, has_red_flag=bool(flagged), flagged_tickers=flagged)

    summary = "**Portfolio 1-Year Returns**\n" + "\n".join(f"- **{t}**: {v:+.2f}%" for t, v in all_returns.items())
    if flagged:
        summary += f"\n\n⚠️ Red flags: {', '.join(flagged)}"
    return StepOutput(step_name="fetch_portfolio", content=summary, success=True)


async def fetch_watchlist(_, session_state):
    watchlist = session_state.get("watchlist_tickers", ["TSMC", "META", "MSFT", "GOOGL"])
    resp = await market_research_agent.arun(f"Get key fundamentals for: {', '.join(watchlist)}. Brief bullet per ticker.")
    return StepOutput(step_name="fetch_watchlist", content=to_str(resp), success=True)


async def fetch_macro(_, session_state):
    resp = await risk_analyst_agent.arun("Calculate the annualized daily return volatility for SPY over the last year.")
    content_str = to_str(resp)

    vol = None
    c = resp.content if resp else None
    if c is not None and hasattr(c, "risk_analyst"):
        lv = c.risk_analyst.last_vol
        if isinstance(lv, dict):
            vol = lv.get("annualized_vol_pct")
    if vol is None:
        m = re.search(r"annualized_vol_pct['\"]?\s*[:=]\s*([\d.]+)", content_str)
        if m:
            vol = float(m.group(1))

    circuit_breaker = vol is not None and vol > 25.0
    session_state.update(circuit_breaker=circuit_breaker, macro_content=content_str, spy_vol=vol)
    regime = "⚠️ HIGH VOLATILITY — circuit breaker active" if circuit_breaker else "✅ Normal regime"
    summary = f"**SPY Annualized Vol:** {vol:.2f}% — {regime}" if vol is not None else content_str
    return StepOutput(step_name="fetch_macro", content=summary, success=True)


async def early_stop(_, session_state):
    macro = session_state.get("macro_content", "Extreme volatility detected.")
    briefing = (
        "# ⚠️ Morning Briefing — Circuit Breaker Active\n\n"
        "Market volatility is extreme. No new positions recommended today.\n\n"
        f"**Macro Signal:**\n{macro}"
    )
    return StepOutput(step_name="early_stop", content=briefing, success=True, stop=True)

async def continue_pipeline(_):
    return StepOutput(step_name="continue", content="Proceeding to full research pipeline.", success=True)

def circuit_breaker_selector(_, session_state):
    if session_state.get("circuit_breaker", False):
        return "early_stop"
    return "continue_pipeline"


async def per_ticker_research(_, session_state):
    tickers = session_state.get("portfolio_tickers", [])
    idx = session_state.get("research_idx", 0)

    if idx >= len(tickers):
        return StepOutput(step_name="per_ticker_research", content="RESEARCH_COMPLETE", success=True)

    ticker = tickers[idx]
    resp = await market_research_agent.arun(
        f"For {ticker}: 6-month price history and latest analyst recommendations. 3-sentence summary."
    )
    results = session_state.get("ticker_research", {})
    results[ticker] = to_str(resp)
    new_idx = idx + 1
    session_state.update(ticker_research=results, research_idx=new_idx)
    content = "RESEARCH_COMPLETE" if new_idx >= len(tickers) else f"Researched {ticker} ({new_idx}/{len(tickers)})"
    return StepOutput(step_name="per_ticker_research", content=content, success=True)

def research_done(outputs):
    return bool(outputs) and "RESEARCH_COMPLETE" in str(outputs[-1].content)


async def red_flag_evaluator(_, session_state):
    return session_state.get("has_red_flag", False)

async def risk_deep_dive(_, session_state):
    flagged = session_state.get("flagged_tickers") or session_state.get("portfolio_tickers", ["SPY"])
    resp = await risk_analyst_agent.arun(f"Max drawdown and SPY correlation for: {', '.join(flagged)}.")
    content = to_str(resp)
    session_state["risk_content"] = content
    return StepOutput(step_name="risk_deep_dive", content=content, success=True)


async def discord_notify(step_input: StepInput) -> StepOutput:
    content = step_input.get_step_content("reporter")
    if not content:
        return StepOutput(step_name="discord_notify", content="Nothing to send.", success=True)
    channel_id = os.getenv("DISCORD_CHANNEL_ID")
    bot_token = os.getenv("DISCORD_BOT_TOKEN")
    if not channel_id or not bot_token:
        return StepOutput(step_name="discord_notify", content="Discord env vars not set, skipping.", success=True)
    agent = Agent(
        tools=[DiscordTools(bot_token=bot_token, enable_send_message=True)],
        model=OpenAIResponses(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
    )
    await agent.arun(f"Send this message to Discord channel {channel_id}:\n\n{content}")
    return StepOutput(step_name="discord_notify", content="Sent to Discord.", success=True)


async def reporter(step_input, session_state):
    ticker_research = session_state.get("ticker_research", {})
    research_md = "\n\n".join(f"### {t}\n{v}" for t, v in ticker_research.items()) or "N/A"

    prompt = (
        "Write a concise morning market briefing in Markdown using the data below. "
        "End with a short 'Key Actions' bullet list (informational only).\n\n"
        f"## Portfolio Health\n{step_input.get_step_content('fetch_portfolio') or 'N/A'}\n\n"
        f"## Watchlist\n{step_input.get_step_content('fetch_watchlist') or 'N/A'}\n\n"
        f"## Macro Regime\n{step_input.get_step_content('fetch_macro') or 'N/A'}\n\n"
        f"## Per-Ticker Research\n{research_md}\n\n"
        f"## Risk Analysis\n{session_state.get('risk_content', 'No red flags detected.')}"
    )
    resp = await coordinate_team.arun(prompt)
    return StepOutput(step_name="reporter", content=to_str(resp), success=True)


async def mm_ingest_pdf(_, session_state):
    file_path = session_state.get("pdf_path")
    if not file_path:
        return StepOutput(step_name="mm_ingest_pdf", content="No pdf_path in session_state.", success=True)
    await multimodal_agent.arun(
        f"Ingest this 10-K filing: {file_path}",
        session_state=session_state,
    )
    return StepOutput(step_name="mm_ingest_pdf", content=str(session_state.get("last_ingest")), success=True)


async def mm_transcribe_audio(_, session_state):
    audio_path = session_state.get("audio_path")
    if not audio_path:
        return StepOutput(step_name="mm_transcribe_audio", content="No audio_path in session_state.", success=True)
    ticker = session_state.get("audio_ticker", "UNKNOWN")
    await multimodal_agent.arun(
        f"Transcribe this earnings call audio for {ticker}: {audio_path}",
        session_state=session_state,
    )
    return StepOutput(step_name="mm_transcribe_audio", content=str(session_state.get("transcript")), success=True)


async def mm_generate_chart(_, session_state):
    ticker = session_state.get("chart_ticker", "SPY")
    range_ = session_state.get("chart_range", "1y")
    await multimodal_agent.arun(
        f"Generate a price chart for {ticker} over {range_}",
        session_state=session_state,
    )
    return StepOutput(step_name="mm_generate_chart", content=str(session_state.get("last_chart")), success=True)


async def mm_describe_screenshot(_, session_state):
    image_path = session_state.get("screenshot_path")
    if not image_path:
        return StepOutput(step_name="mm_describe_screenshot", content="No screenshot_path in session_state.", success=True)
    label = session_state.get("screenshot_label", "")
    await multimodal_agent.arun(
        f"Describe this dashboard screenshot. Label: {label}. Path: {image_path}",
        session_state=session_state,
    )
    return StepOutput(step_name="mm_describe_screenshot", content=str(session_state.get("last_description")), success=True)


multimodal_workflow = Workflow(
    id="multimodal",
    name="multimodal",
    db=db,
    session_state={},
    steps=[
        mm_ingest_pdf,
        mm_transcribe_audio,
        mm_generate_chart,
        mm_describe_screenshot,
    ],
)


def _morning_session_state():
    return {
        "portfolio_tickers": ["NVDA", "AAPL", "BND"],
        "watchlist_tickers": ["TSMC", "META", "MSFT", "GOOGL"],
        "circuit_breaker": False,
        "has_red_flag": False,
        "flagged_tickers": [],
        "ticker_research": {},
        "research_idx": 0,
        "portfolio_returns": {},
        "spy_vol": None,
        "macro_content": "",
        "risk_content": "",
    }


morning_briefing_workflow = Workflow(
    id="morning_briefing",
    name="morning_briefing",
    db=db,
    session_state=_morning_session_state(),
    steps=[
        Parallel(fetch_portfolio, fetch_watchlist, fetch_macro, name="parallel_fetch"),
        Router(choices=[early_stop, continue_pipeline], selector=circuit_breaker_selector, name="circuit_breaker_router"),
        Loop(steps=[per_ticker_research], max_iterations=10, end_condition=research_done, name="ticker_research_loop"),
        Condition(steps=[risk_deep_dive], evaluator=red_flag_evaluator, name="risk_condition"),
        reporter,
        discord_notify,
    ],
)


async def trade_step(step_input: StepInput, session_state: dict) -> StepOutput:
    resp = await trade_agent.arun(step_input.input)
    content = resp.content if resp else None
    if content is None:
        return StepOutput(step_name="trade_step", content="No response from trade agent.", success=False)
    text = content if isinstance(content, str) else str(content)
    return StepOutput(step_name="trade_step", content=text, success=True)


trade_workflow = Workflow(
    id="trade_workflow",
    name="trade_workflow",
    db=db,
    session_state={"permitted_tickers": None},
    steps=[
        trade_step,
    ],
)


BT_TERMINAL = {"COMPLETED", "CANCELLED", "ERROR"}


async def launch_backtest(step_input: StepInput, session_state: dict) -> StepOutput:
    strategy = session_state.get("strategy") or step_input.input or "AAPL,MSFT,NVDA"
    range_ = session_state.get("range", "1y")

    run_output = await backtest_agent.arun(
        f"Backtest this strategy over {range_}: {strategy}",
        background=True,
    )

    session_state["bt_run_id"] = run_output.run_id
    session_state["bt_session_id"] = run_output.session_id
    return StepOutput(
        step_name="launch_backtest",
        content=f"LAUNCHED run_id={run_output.run_id}",
        success=True,
    )


async def poll_backtest(step_input: StepInput, session_state: dict) -> StepOutput:
    run_id = session_state.get("bt_run_id")
    session_id = session_state.get("bt_session_id")

    await asyncio.sleep(3)

    result = await backtest_agent.aget_run_output(run_id, session_id=session_id)
    status = result.status.value if result else "ERROR"

    if status in BT_TERMINAL:
        session_state["bt_result"] = result.content if result else None
        return StepOutput(step_name="poll_backtest", content=f"POLL_DONE status={status}", success=True)

    return StepOutput(step_name="poll_backtest", content=f"POLL_RUNNING status={status}", success=True)


def backtest_poll_done(outputs: list) -> bool:
    return bool(outputs) and "POLL_DONE" in str(outputs[-1].content)


async def finalize_backtest(step_input: StepInput, session_state: dict) -> StepOutput:
    run_id = session_state.get("bt_run_id", "unknown")
    result = session_state.get("bt_result")
    return StepOutput(step_name="finalize_backtest", content=f"Backtest {run_id} complete.\n\n{result}", success=True)


backtest_workflow = Workflow(
    id="backtest_workflow",
    name="backtest_workflow",
    db=db,
    session_state={"strategy": None, "range": "1y", "bt_run_id": None, "bt_session_id": None, "bt_result": None},
    steps=[
        launch_backtest,
        Loop(steps=[poll_backtest], max_iterations=20, end_condition=backtest_poll_done, name="poll_loop"),
        finalize_backtest,
    ],
)
