import asyncio
import io
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf

from agno.agent import Agent
from agno.media import Image
from agno.models.openai import OpenAIResponses
from agno.run import RunContext
from agno.tools import tool
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from agents.market_research_agent import db, knowledge_base, maybe_embed_runtime_kb_payload
from models.schema import (
    ChartOutput,
    DescriptionOutput,
    IngestOutput,
    MultimodalAgentOutput,
    MultimodalContent,
    MultimodalSessionInput,
    TranscriptOutput,
)

load_dotenv()


@tool
async def ingest_pdf_filing(
    run_context: RunContext,
    file_path: str,
    ticker: str,
    year: int,
    doc_type: str = "10-K",
):
    path = Path(file_path).resolve()
    if not path.exists():
        result = IngestOutput(error=f"File not found: {file_path}")
        run_context.session_state["last_ingest"] = result.model_dump()
        return result

    await knowledge_base.ainsert(
        name=path.name,
        path=str(path),
        metadata={"ticker": ticker.upper(), "year": year, "doc_type": doc_type},
        skip_if_exists=True,
    )
    result = IngestOutput(file=path.name, ticker=ticker.upper(), year=year, doc_type=doc_type)
    run_context.session_state["last_ingest"] = result.model_dump()
    return result


@tool
async def transcribe_earnings_call(
    run_context: RunContext,
    audio_path: str,
    ticker: str,
    source: str = "earnings_call",
):
    import openai

    path = Path(audio_path).resolve()
    if not path.exists():
        result = TranscriptOutput(error=f"File not found: {audio_path}")
        run_context.session_state["transcript"] = result.model_dump()
        return result

    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    with open(path, "rb") as f:
        response = await client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text",
        )

    transcript = response if isinstance(response, str) else response.text
    result = TranscriptOutput(ticker=ticker.upper(), text=transcript, source=source)
    run_context.session_state["transcript"] = result.model_dump()

    await maybe_embed_runtime_kb_payload(
        kb_name=f"runtime:transcript:{ticker.upper()}:{source}",
        description=f"Earnings call transcript for {ticker.upper()}",
        payload={"transcript": transcript},
        metadata={"source": source, "kind": "transcript", "ticker": ticker.upper()},
    )
    return result


@tool
async def generate_price_chart(
    run_context: RunContext,
    ticker: str,
    range: str = "1y",
):
    t = ticker.strip().upper()
    data = await asyncio.to_thread(yf.download, t, period=range, interval="1d", progress=False)

    if data is None or data.empty:
        result = ChartOutput(error="no data", ticker=t)
        run_context.session_state["last_chart"] = result.model_dump()
        return result

    close = data["Close"]
    if isinstance(close, type(data)):
        close = close.iloc[:, 0]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(close.index, close.values, linewidth=1.5, color="#2563eb")
    ax.set_title(f"{t} — {range}", fontsize=13)
    ax.set_ylabel("Price (USD)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    buf.seek(0)
    plt.close(fig)

    image_bytes = buf.getvalue()
    result = ChartOutput(ticker=t, range=range, bytes_len=len(image_bytes))
    run_context.session_state["last_chart"] = result.model_dump()
    return result


@tool
async def describe_dashboard_screenshot(
    run_context: RunContext,
    image_path: str,
    label: str = "",
):
    path = Path(image_path).resolve()
    if not path.exists():
        result = DescriptionOutput(error=f"File not found: {image_path}")
        run_context.session_state["last_description"] = result.model_dump()
        return result

    image_bytes = path.read_bytes()

    vision_agent = Agent(
        model=OpenAIResponses(id="gpt-4o", api_key=os.getenv("OPENAI_API_KEY")),
        markdown=False,
    )
    prompt = f"Describe this dashboard screenshot in detail. Label: {label}" if label else "Describe this dashboard screenshot in detail."
    resp = await vision_agent.arun(prompt, images=[Image(content=image_bytes)])
    description = resp.content if isinstance(resp.content, str) else str(resp.content)

    result = DescriptionOutput(image=path.name, label=label, description=description)
    run_context.session_state["last_description"] = result.model_dump()

    await maybe_embed_runtime_kb_payload(
        kb_name=f"runtime:screenshot:{path.stem}",
        description=f"Dashboard screenshot description: {label or path.name}",
        payload={"description": description, "label": label},
        metadata={"kind": "screenshot_description", "file": path.name, "label": label},
    )
    return result


async def merge_multimodal_output_from_session_state(run_output, run_context, **_kwargs):
    state = run_context.session_state or {}
    last_ingest = state.get("last_ingest")
    transcript = state.get("transcript")
    last_chart = state.get("last_chart")
    last_description = state.get("last_description")
    run_output.content = MultimodalAgentOutput(
        multimodal=MultimodalContent(
            last_ingest=IngestOutput(**last_ingest) if isinstance(last_ingest, dict) else last_ingest,
            transcript=TranscriptOutput(**transcript) if isinstance(transcript, dict) else transcript,
            last_chart=ChartOutput(**last_chart) if isinstance(last_chart, dict) else last_chart,
            last_description=DescriptionOutput(**last_description) if isinstance(last_description, dict) else last_description,
        )
    )


def multimodal_tools_factory(agent=None, run_context=None):
    return [ingest_pdf_filing, transcribe_earnings_call, generate_price_chart, describe_dashboard_screenshot]


def create_multimodal_agent(team=None, run_context=None):
    return Agent(
        db=db,
        name="multimodal_agent",
        description="Ingests PDFs, transcribes audio, generates price charts, and describes screenshots.",
        instructions="Process the provided file or media using the appropriate tool (ingest PDF, transcribe audio, chart price, or describe screenshot) and return a structured summary of the output.",
        model=OpenAIResponses(id="gpt-4o", api_key=os.getenv("OPENAI_API_KEY")),
        tools=multimodal_tools_factory,
        add_session_state_to_context=True,
        session_state=MultimodalSessionInput().model_dump(),
        search_knowledge=True,
        add_history_to_context=False,
        output_schema=MultimodalAgentOutput,
        post_hooks=[merge_multimodal_output_from_session_state],
        markdown=True,
    )


multimodal_agent = create_multimodal_agent()
