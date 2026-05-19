import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools import tool
from agno.run import RunContext
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from agents.market_research_agent import db, knowledge_base
from models.schema import GeneralEnquiryAgentOutput, GeneralEnquiryContent

load_dotenv()


@tool
async def utc_now(run_context: RunContext):
    now = datetime.now(timezone.utc).isoformat()
    run_context.session_state["last_utc"] = now
    return GeneralEnquiryContent(last_utc=now)


@tool
async def capabilities_blurb(run_context: RunContext):
    txt = (
        "This assistant can combine live tools with the knowledge base (PDFs) for market and finance Q&A. "
        "It does not execute trades or give personalized investment advice."
    )
    run_context.session_state["capabilities"] = txt
    return GeneralEnquiryContent(capabilities=txt)


@tool
async def normalize_question(run_context: RunContext, text: str, max_chars: int = 500):
    q = " ".join(text.split()).strip()
    if len(q) > max_chars:
        q = q[: max_chars - 3] + "..."
    run_context.session_state["normalized_question"] = q
    return GeneralEnquiryContent(normalized_question=q)


def general_enquiry_tools_factory(agent=None, run_context=None):
    return [utc_now, capabilities_blurb, normalize_question]


async def merge_general_enquiry_output_from_session_state(run_output, run_context, **_kwargs):
    state = run_context.session_state or {}
    # Preserve text_response from the LLM's structured output before overwriting content.
    # When no tool is called the LLM's knowledge answer lives only in text_response;
    # without this it is silently discarded.
    llm_content = run_output.content
    text_response = None
    if isinstance(llm_content, GeneralEnquiryAgentOutput):
        text_response = llm_content.general_enquiry.text_response
    run_output.content = GeneralEnquiryAgentOutput(
        general_enquiry=GeneralEnquiryContent(
            last_utc=state.get("last_utc"),
            capabilities=state.get("capabilities"),
            normalized_question=state.get("normalized_question"),
            text_response=text_response,
        )
    )


def create_general_enquiry_agent(team=None, run_context=None, structured_output=True):
    kwargs = dict(
        db=db,
        knowledge=knowledge_base,
        search_knowledge=True,
        add_knowledge_to_context=True,
        name="general_enquiry_agent",
        description="General finance questions using KB search and light utilities.",
        instructions=(
            "Answer general finance questions concisely by searching the knowledge base first, "
            "then supplement with available utilities; do not execute trades or give personalized "
            "investment advice. Always populate the text_response field of your output with your "
            "complete answer — this is especially important when no tool is called and the answer "
            "comes from your training knowledge."
        ),
        reasoning=True,
        tools=general_enquiry_tools_factory,
        add_history_to_context=True,
        add_session_state_to_context=True,
        session_state={"last_utc": None, "capabilities": None, "normalized_question": None},
        model=OpenAIResponses(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
    )
    if structured_output:
        kwargs["output_schema"] = GeneralEnquiryAgentOutput
        kwargs["post_hooks"] = [merge_general_enquiry_output_from_session_state]
    return Agent(**kwargs)


general_enquiry_agent = create_general_enquiry_agent()
