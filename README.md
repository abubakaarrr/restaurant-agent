# Restaurant AI Receptionist — POC

An AI phone receptionist for **La Casa Restaurant**, built with:
- **LangGraph** — stateful agent orchestration (ReAct pattern)
- **Claude claude-sonnet-4-20250514** (Anthropic) — the LLM
- **FastAPI** — REST API + Twilio webhook (future)
- **PostgreSQL + pgvector** — live bookings + RAG knowledge base
- **OpenAI text-embedding-3-small** — knowledge chunk embeddings

---

## Project Structure

```
restaurant-agent/
├── app/
│   ├── main.py                   # FastAPI entry point
│   ├── config.py                 # Settings from .env
│   ├── agent/
│   │   ├── graph.py              # LangGraph graph (START → agent ↔ tools → END)
│   │   ├── state.py              # RestaurantAgentState
│   │   ├── configuration.py      # Runtime config (model, restaurant name)
│   │   └── nodes/
│   │       ├── generate_response.py   # Claude node with bound tools
│   │       └── tool_executor.py       # LangGraph ToolNode
│   ├── tools/
│   │   ├── rag.py                # pgvector semantic search tools
│   │   └── db.py                 # Booking + availability tools
│   ├── prompts/
│   │   └── system.md             # Agent persona + rules
│   └── knowledge/
│       ├── menu.md               # Full restaurant menu (embedded)
│       ├── slots.md              # Hours + table config (embedded)
│       └── restaurant_info.md    # FAQs + location + policies (embedded)
├── db/
│   ├── schema.sql                # PostgreSQL + pgvector schema
│   └── seed.py                   # Seed tables, menu items, embed knowledge files
└── tests/
    └── simulate_call.py          # Demo: run 3 call scenarios without Twilio
```

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- PostgreSQL 15+ with [pgvector extension](https://github.com/pgvector/pgvector)
- API keys: Anthropic (Claude) + OpenAI (embeddings)

### 2. Install

```bash
cd restaurant-agent
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env — fill in DATABASE_URL, ANTHROPIC_API_KEY, OPENAI_API_KEY
```

### 4. Set up database

```bash
createdb restaurant_agent
psql restaurant_agent -f db/schema.sql
python db/seed.py
```

### 5. Run the demo simulation (no Twilio needed)

```bash
python tests/simulate_call.py
```

### 6. Start the API server

```bash
uvicorn app.main:app --reload
# API docs: http://localhost:8000/docs
```

---

## Agent Flow

```
Caller message
      │
      ▼
  [agent node]  ←──────────────────────────┐
  Claude decides:                           │
    • needs menu info?  → search_menu       │
    • needs table check? → check_table_availability
    • ready to book?    → create_booking    │
    • general question? → search_restaurant_info
    • has all info?     → respond directly  │
      │                                     │
      ▼ (tool call present)                 │
  [tool node]  ─────────────────────────────┘
  Executes tool, returns result
      │
      ▼ (no more tool calls)
    END → voice response (≤40 words)
```

---

## Tools Available to the Agent

| Tool | Purpose |
|------|---------|
| `search_menu` | RAG search on menu.md — answers dietary, price, ingredient questions |
| `search_restaurant_info` | RAG search on info + slots — answers hours, location, parking, policies |
| `check_table_availability` | Live SQL query — checks if tables are free for given date/time/party |
| `create_booking` | Atomic SQL insert — saves confirmed booking with row-level lock |
| `check_menu_item_availability` | Live SQL query — checks if a specific dish is sold out |

---

## Extending the POC

| What to add | Where |
|-------------|-------|
| Twilio voice integration | `app/telephony/twilio_handler.py` (not in POC scope) |
| Order tracking | New tool in `app/tools/db.py` + update schema |
| Human handoff | Add `end_call` condition in `graph.py` + `<Dial>` in TwiML |
| Real-time specials | Update `menu_items` table; agent reads `check_menu_item_availability` |
| LangSmith tracing | Set `LANGCHAIN_TRACING_V2=true` in `.env` |
