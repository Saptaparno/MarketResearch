import asyncio
from agno.utils.pprint import pprint_run_response
from core.workflow import morning_briefing_workflow, advisory_workflow
from dotenv import load_dotenv

load_dotenv()


from agents.market_research_agent import (
    KNOWLEDGE_BASE_DIR,
    knowledge_base,
    knowledgebase_pg_url,
    vectordb,
)



async def verify_knowledge_base(sample_query: str = "market data pricing history") -> None:
    """Report vector row count and a short async search smoke test."""
    row_count = vectordb.get_count()
    print(f"Knowledge base vector rows: {row_count}")
    docs = await knowledge_base.asearch(sample_query, max_results=3)
    print(f"Sample asearch({sample_query!r}) -> {len(docs)} document(s)")
    for i, doc in enumerate(docs):
        label = getattr(doc, "name", None) or getattr(doc, "id", "?")
        content = (getattr(doc, "content", None) or "").replace("\n", " ")
        preview = f"{content[:160]}…" if len(content) > 160 else content
        print(f"  [{i}] {label}: {preview}")


async def main() -> None:
    if knowledgebase_pg_url and KNOWLEDGE_BASE_DIR.is_dir():
        for pdf in sorted(KNOWLEDGE_BASE_DIR.glob("*.pdf")):
            await knowledge_base.ainsert(
                name=pdf.name,
                path=str(pdf.resolve()),
                skip_if_exists=True,
            )
    await verify_knowledge_base()

    morning_briefing_prompt = (
        "What are the latest news and events that are happening in the market?"
    )

    morning_briefing_result = await morning_briefing_workflow.arun(input=morning_briefing_prompt)
    pprint_run_response(morning_briefing_result)


if __name__ == "__main__":
    asyncio.run(main())
