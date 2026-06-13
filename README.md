# Valura — Multi-Agent Market Intelligence (Agno)

A multi-agent market research and advisory platform built on [Agno](https://docs.agno.com). It combines specialized agents, coordinated teams, deterministic workflows, PgVector-backed knowledge, human-in-the-loop trade safety, and an AgentOS runtime with scheduling and tracing.

This repo implements the **Valura Agno Mastery Assignment** (`Valura_Agno_Assignment.pdf`): a training simulator for finance workflows — not a production trading system.

## Features

- **Agents** — Market research, portfolio health, risk, financial planning, product recommendations, general enquiry, trade (HITL), multimodal (PDF/audio/charts), and backtest.
- **Teams** — `coordinate`, `route`, and `broadcast` modes over a shared advisory member set.
- **Workflows** — Morning briefing (parallel fetch → circuit breaker → per-ticker loop → risk → reporter → Discord), advisory pipeline, trade, multimodal, and background backtest polling.
- **Knowledge** — PgVector store with PDF ingestion from `knowledge_base/`.
- **Persistence** — Postgres sessions, memory, traces, and scheduler tables via `AsyncPostgresDb`.
- **Safety** — PII masking, prompt-injection blocking, ticker permission guardrail, tiered trade approvals. See [SECURITY.md](SECURITY.md).
- **Evals** — Accuracy, reliability, performance, and agent-as-judge suites under `tests/evals/`.

## Prerequisites

- Python 3.12+
- Docker Desktop (for Postgres + pgvector)
- OpenAI API key

## Quick start

### 1. Start databases

```bash
docker compose up -d
```

| Service | Host port | Purpose |
| -------- | ---------- | -------- |
| `postgres` | `5433` | AgentOS sessions, memory, traces, schedules |
| `postgres_knowledgebase` | `5434` | PgVector knowledge base |

### 2. Configure environment

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...

DATABASE_URL=postgresql://market:market@localhost:5433/marketresearch
KNOWLEDGEBASE_DATABASE_URL=postgresql://kb:kb@localhost:5434/knowledgebase

# Optional
KNOWLEDGE_BASE_DIR=./knowledge_base
DISCORD_BOT_TOKEN=
DISCORD_CHANNEL_ID=
```

`DATABASE_URL` uses the async driver internally (`postgresql+asyncpg://…`). `KNOWLEDGEBASE_DATABASE_URL` is normalized for sync PgVector (`postgresql+psycopg://…`).

### 3. Install dependencies

Install Agno and project dependencies (adjust to your environment):

```bash
pip install agno python-dotenv uvicorn yfinance pandas matplotlib openai httpx discord.py
```

Use a Postgres driver stack compatible with Agno (`asyncpg`, `psycopg` / `psycopg2-binary`) as required by your Agno version.

### 4. Run AgentOS

```bash
python AgentOS.py
```

The API listens on **http://0.0.0.0:8000**. AgentOS registers all agents, teams, workflows, the knowledge base, OpenTelemetry tracing, and an in-process **scheduler** (polls every 15s).

Connect the [Agno Control Plane](https://os.agno.com) to inspect traces, sessions, schedules, and HITL approvals.

### 5. Smoke test (local script)

Seed PDFs into the knowledge base and run the morning briefing workflow once:

```bash
python main.py
```

## Project layout

```
agents/           # Specialist agents (tools, output schemas, hooks)
core/
  workflow.py     # Teams, workflows, step executors
  scheduler.py    # Register cron schedule for morning briefing
  background_jobs.py  # Schedule verification, backtest cancellation helpers
guardrails/       # Custom guardrails (e.g. ticker permissions)
models/           # Pydantic output schemas
knowledge_base/   # PDFs for ingestion
tests/
  evals/          # Phase 10 eval suites
  test_hitl_guardrails.py
docs/             # learnings.md, eval_report.md
AgentOS.py        # AgentOS entrypoint (FastAPI app)
main.py           # Local workflow + KB smoke test
docker-compose.yml
SECURITY.md
sample_inputs.txt # Example prompts per workflow
```

## Agents

| Agent | Role |
| ----- | ---- |
| `market_research_agent` | Yahoo Finance data, analyst recs, knowledge search |
| `portfolio_health_agent` | Quotes, returns, comparisons |
| `risk_analyst_agent` | Volatility, drawdown, SPY correlation |
| `financial_planning_agent` | FV, monthly savings, loan payment calculators |
| `product_recommendation_agent` | Exploratory fund/ETF framing |
| `general_enquiry_agent` | General finance Q&A via knowledge base |
| `trade_agent` | Paper trade proposals with HITL and guardrails |
| `multimodal_agent` | PDF ingest, audio transcription, charts, screenshots |
| `backtest_agent` | Long-running strategy backtest (background runs) |

## Teams

| Team | Mode | Use |
| ---- | ---- | --- |
| `coordinate_team` | coordinate | Open-ended multi-specialist tasks |
| `route_team` | route | Single best-fit specialist per message |
| `broadcast_team` | broadcast | Same task to all members, merged reply |

## Workflows

| Workflow | Description |
| -------- | ----------- |
| `morning_briefing` | Parallel portfolio/watchlist/macro → circuit breaker → ticker loop → conditional risk → reporter → optional Discord post |
| `advisory_workflow` | Coordinate → route → broadcast advisory pipeline |
| `trade_workflow` | Trade agent with session guardrails |
| `multimodal` | PDF, audio, chart, screenshot steps |
| `backtest_workflow` | Launch background backtest → poll until terminal → finalize |

Example inputs and `session_state` overrides are in [sample_inputs.txt](sample_inputs.txt).

## Scheduler

With `scheduler=True` on AgentOS, cron jobs are stored in `agno_schedules` and executed by the built-in poller.

Register the daily morning briefing (6:00 Asia/Kolkata, skips if already exists):

```bash
python core/scheduler.py
```

Verify schedules and recent runs:

```bash
python core/background_jobs.py
```

Morning briefing output can be posted to Discord when `DISCORD_BOT_TOKEN` and `DISCORD_CHANNEL_ID` are set (see `discord_notify` in `core/workflow.py`).

## Evals

Run all four suites (accuracy, reliability, performance, agent-as-judge):

```bash
python tests/evals/run_all.py
```

Results and methodology are documented in [docs/eval_report.md](docs/eval_report.md). Phase notes live in [docs/learnings.md](docs/learnings.md).

## Security

Trade proposals use three tiers of human-in-the-loop control plus input guardrails and a background audit log. Full details: [SECURITY.md](SECURITY.md).

Run guardrail / HITL tests:

```bash
pytest tests/test_hitl_guardrails.py
```

## Disclaimer

Valura is an **Agno learning project**. It uses paper data and simulated trades. Do not use it to manage real money or execute live orders.
