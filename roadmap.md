# NPS LLM Tutor — Project Roadmap

## Project Overview

**Goal:** An agentic LLM tutoring system for SS3861 (Spacecraft Payload Communications & Data Handling) that answers student questions using course materials (lectures, labs, operations manuals), tracks student progress, and is deployable from GitHub via Docker.

**Current state:** Working RAG pipeline + LangGraph agent in `NPS_tutor.ipynb`. ~875 document chunks in ChromaDB (lectures, labs, manual). Evaluation framework in `tutor_evaluations.py`. NPS-local Llama endpoint as LLM backend.

**End state:** Modular Python application with Streamlit UI, SQLite3 user/session storage, cloud LLM backend (Anthropic/OpenAI), Docker deployment, and a comprehensive evaluation suite.

---

## Directory Structure (Target)

```
thesis_llm/
├── agent/
│   └── agent.py              # LangGraph agent, tools, system prompt
├── app/
│   └── app.py                # Streamlit UI (chat, login, dashboard)
├── db/
│   └── database.py           # SQLite3 schema + CRUD helpers
├── eval/
│   └── evaluations.py        # Extended evaluation framework
├── rag/
│   ├── rag.py                # ExampleRAG class (existing, minor edits)
│   └── load_dataset.py       # Vectorstore loading (existing, minor edits)
├── scripts/
│   ├── build_vectorstore.py  # Reproducible ChromaDB build from JSON files
│   └── ingest_manual.py      # [PLACEHOLDER] Satellite ops manual ingestion
├── data/
│   ├── lectures.json         # Existing
│   ├── labs.json             # Existing
│   ├── manual.json           # Existing
│   ├── transcripts.json      # Existing
│   └── eval_questions.json   # Existing (expand to 20+ questions)
├── .env.example              # API key template (never commit .env)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt          # Update with new dependencies
├── NPS_tutor.ipynb           # Keep as development/research scratchpad
└── roadmap.md                # This file
```

---

## Phase 1 — Project Restructuring

**Goal:** Extract the working notebook code into importable Python modules and switch the LLM backend to a cloud API.

**Checkpoint:** `python agent/agent.py` runs a test query end-to-end using the cloud API.

### Steps

1. **Create `agent/agent.py`**
   - Copy the LangGraph agent definition from `NPS_tutor.ipynb` (cells 36–50)
   - Replace `ChatOpenAI(base_url=NPS_LOCAL_URL, ...)` with `ChatAnthropic(model="claude-sonnet-4-6")` or `ChatOpenAI(model="gpt-4o")`
   - Load API key from environment variable via `python-dotenv`
   - Keep the three RAG tools (`semantic_retrieve_lecture_w_scores`, `semantic_retrieve_lab_w_scores`, `semantic_retrieve_manual_w_scores`) and the DuckDuckGo web search tool
   - Accept `user_context` dict param to inject name/role into system prompt
   - Accept `conversation_history` list param to seed `MessagesState`

2. **Create `scripts/build_vectorstore.py`**
   - Replicate the vectorstore creation logic from `NPS_tutor.ipynb` (cells 27–35) as a standalone script
   - Read from `data/lectures.json`, `data/labs.json`, `data/manual.json`
   - Persist to `./chroma_db/` (already used in existing code)
   - This makes the vectorstore reproducible for any user who clones the repo

3. **Create `.env.example`**
   ```
   ANTHROPIC_API_KEY=your_key_here
   # or
   OPENAI_API_KEY=your_key_here
   LLM_PROVIDER=anthropic   # or "openai"
   CHROMA_DB_PATH=./chroma_db
   SQLITE_DB_PATH=./tutor.db
   ```

4. **Update `requirements.txt`**
   - Add: `anthropic>=0.75.0`, `streamlit>=1.35.0`, `python-dotenv>=1.0.0`, `streamlit-chat`, `plotly>=5.0.0`
   - Confirm `langchain-anthropic` is included for LangGraph tool-calling support

5. **Minimal edits to `rag/rag.py` and `rag/load_dataset.py`**
   - Update `setup_embedding_function()` in `load_dataset.py`: remove hard-coded NPS server URL; fall back directly to local `all-MiniLM-L6-v2` or accept an embedding server URL from env var
   - No other changes needed to existing RAG code

---

## Phase 2 — SQLite3 Database Layer

**Goal:** Persistent storage for user identities, conversation history, and evaluation scores so the agent knows who it is talking to across sessions.

**Checkpoint:** Python unit test confirms create/read/update operations on all four tables.

### Schema

```sql
CREATE TABLE users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    role        TEXT NOT NULL CHECK(role IN ('student', 'instructor')),
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id),
    started_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE messages (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id  INTEGER NOT NULL REFERENCES conversations(id),
    role             TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'tool')),
    content          TEXT NOT NULL,
    timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE eval_scores (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL REFERENCES users(id),
    question_id  TEXT NOT NULL,
    scores       TEXT NOT NULL,   -- JSON: {correctness, reasoning, pedagogy, grounding, overall}
    timestamp    DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE tool_usage_logs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id  INTEGER NOT NULL REFERENCES conversations(id),
    message_id       INTEGER REFERENCES messages(id),
    tool_name        TEXT NOT NULL,
    query            TEXT,
    retrieved_docs   TEXT,  -- JSON array of {source, score} pairs
    timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Steps

1. **Create `db/database.py`** with the following functions:
   - `init_db(db_path)` — create all tables if they don't exist
   - `create_user(name, role)` → `user_id`
   - `get_user(user_id)` → user dict
   - `get_or_create_user(name, role)` → user dict (for Streamlit login)
   - `start_conversation(user_id)` → `conversation_id`
   - `save_message(conversation_id, role, content)` → `message_id`
   - `get_conversation_history(conversation_id, limit=20)` → list of `{role, content}` dicts
   - `save_eval_score(user_id, question_id, scores_dict)`
   - `get_user_eval_scores(user_id)` → list of score records
   - `log_tool_usage(conversation_id, message_id, tool_name, query, retrieved_docs)`
   - `get_tool_usage_stats(conversation_id)` → summary dict

2. **Wire database init into app startup** — call `init_db()` on Streamlit app launch

---

## Phase 3 — Agent Enhancement

**Goal:** The agent uses the logged-in user's profile and prior conversation history to give personalized, context-aware responses.

**Checkpoint:** Sending two messages in sequence — the agent correctly references the student's name and recalls the prior exchange.

### Steps

1. **Personalized system prompt in `agent/agent.py`**
   - Replace static system prompt with a template that injects user name and role:
     ```
     You are an expert tutor for SS3861 (Spacecraft Payload Communications & Data Handling) at NPS.
     You are speaking with {name}, a {role} in the course.
     Use their prior questions to build on what they already know...
     ```
   - If role is `instructor`, shift tone to peer-level technical discussion

2. **Conversation memory injection**
   - Before invoking the agent, retrieve the last 20 messages from `db/database.py:get_conversation_history()`
   - Convert to LangChain `HumanMessage`/`AIMessage` objects and prepend to `MessagesState`
   - After each agent response, persist new messages via `db/database.py:save_message()`

3. **Tool usage logging hook**
   - After each tool call, invoke `db/database.py:log_tool_usage()` with the tool name, query, and retrieved doc sources/scores from the LangGraph state

4. **Expand `data/eval_questions.json`**
   - Grow from 14 to 20+ questions covering all 10 lecture topics:
     - Number systems & binary (existing)
     - Encoding / ASCII (existing)
     - C&DH intro + comm protocols
     - Ethernet / UDP
     - Basic circuits
     - Computer architecture & failure mitigation
     - EPS (Electrical Power System)
     - Spacecraft buses & ICDs
     - Testing procedures
     - Systems engineering & PDR process

5. **[PLACEHOLDER] Satellite Operations Manual Ingestion**
   - Create `scripts/ingest_manual.py` mirroring the docling-based ingestion pattern from `NPS_tutor.ipynb` (cells 8–26)
   - Pipeline: load PDF → docling extraction → semantic chunking → append to `manual.json` → rebuild vectorstore
   - Mark file with `# TODO: Run when satellite ops manual PDF is acquired`
   - The `semantic_retrieve_manual_w_scores` tool in `agent/agent.py` is already wired to use this data

---

## Phase 4 — Evaluation Framework Enhancement

**Goal:** Evaluate both response quality (existing) and agent behavior (tool selection, retrieval quality), and persist all results to SQLite3.

**Checkpoint:** Running `eval/evaluations.py` on 20 test questions produces a per-question score table and stores results in `eval_scores` and `tool_usage_logs`.

### Steps

1. **Persist scores in `eval/evaluations.py`**
   - Wrap `evaluate_batch()` from `tutor_evaluations.py` to call `db/database.py:save_eval_score()` after each question
   - Accept an optional `user_id` param; if None, store under a synthetic "eval_run" user

2. **Tool-usage evaluation metrics**
   - After each agent invocation, check which tools were actually called vs. which were expected (add `expected_tools` field to `eval_questions.json`)
   - Compute:
     - **Tool precision:** fraction of tools called that were appropriate
     - **Tool recall:** fraction of expected tools that were called
     - **Retrieval score:** mean similarity score of top-3 retrieved docs
   - Add these to the per-question score dict

3. **RAGAS integration (optional backend eval)**
   - For RAG-grounded questions, run RAGAS metrics post-hoc:
     - `faithfulness` — is the answer supported by retrieved context?
     - `answer_relevance` — does the answer address the question?
     - `context_precision` — are retrieved docs actually useful?
   - Gate behind `ENABLE_RAGAS=true` env var (RAGAS requires cloud LLM calls, which have cost)

4. **Aggregate reporting function**
   - `generate_eval_report(user_id=None)` in `eval/evaluations.py`
   - Outputs: per-topic average score, tool utilization rate by tool, score trend over time (if user_id provided)
   - Used by the Streamlit instructor dashboard

---

## Phase 5 — Streamlit UI

**Goal:** A browser-based interface that students and instructors can use without touching the command line.

**Checkpoint:** A user can log in, send three messages to the agent, view their conversation history, and see their quiz scores on the dashboard.

### Steps

1. **Login / registration page (`app/app.py`)**
   - On first visit: form with Name and Role (student / instructor) → `db:create_user()` → store `user_id` in `st.session_state`
   - On return visit: name lookup → resume existing user
   - No passwords needed (academic tool, NPS network deployment)

2. **Chat interface**
   - Use `st.chat_message` and `st.chat_input` (native Streamlit chat components)
   - Display conversation history loaded from `db:get_conversation_history()` on page load
   - On submit:
     1. Save user message to DB
     2. Call `agent.invoke(messages, user_context)` with history
     3. Stream response token-by-token using `st.write_stream` if cloud API supports streaming
     4. Save assistant response to DB
   - Show tool calls as expandable info boxes (e.g., `> Used: semantic_retrieve_lecture [score: 0.87]`)

3. **Sidebar**
   - Current user name and role
   - "New conversation" button (starts fresh `conversation_id`)
   - List of past conversation starters (first user message of each session)

4. **Student dashboard tab**
   - Evaluation score history table (question ID, topic, scores, date)
   - Bar chart: average score per topic (uses `plotly`)
   - "Run self-evaluation" button — runs `evaluate_batch()` on a random subset of `eval_questions.json`

5. **Instructor dashboard tab** (only visible when `role == 'instructor'`)
   - Aggregate class view: average scores per topic across all students
   - Tool usage heatmap: which tools are used most per topic
   - Export to CSV button

---

## Phase 6 — Docker Containerization

**Goal:** Anyone who clones the repo can run the full application with a single command.

**Checkpoint:** `docker compose up` on a clean machine brings up the Streamlit UI at `localhost:8501`.

### Steps

1. **`Dockerfile`**
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   EXPOSE 8501
   CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **`docker-compose.yml`**
   ```yaml
   services:
     tutor:
       build: .
       ports:
         - "8501:8501"
       volumes:
         - ./chroma_db:/app/chroma_db     # ChromaDB persistence
         - ./tutor.db:/app/tutor.db        # SQLite3 persistence
       env_file:
         - .env
   ```

3. **Pre-built vectorstore**
   - The `chroma_db/` directory must either be committed (if size allows) or regenerated at first run
   - Add a startup check in `app/app.py`: if `chroma_db/` is empty, run `scripts/build_vectorstore.py` automatically before launching agent
   - Document this in README

4. **`.dockerignore`**
   - Exclude: `.env`, `__pycache__`, `*.pyc`, `Lectures/` (large PPTX/PDF), `Labs/`

---

## Phase 7 — GitHub Deployment Prep

**Goal:** The GitHub repo is the authoritative source for deploying the application. New users can get running in under 10 minutes.

**Checkpoint:** A reviewer with no prior context can follow README.md and reach the Streamlit UI.

### Steps

1. **`README.md` quick-start section**
   ```
   ## Quick Start
   1. Clone this repo
   2. Copy `.env.example` to `.env` and add your Anthropic or OpenAI API key
   3. Run: docker compose up
   4. Open http://localhost:8501
   ```

2. **`scripts/build_vectorstore.py`**
   - Ensure it is idempotent (skip chunks already in ChromaDB by checking existing IDs)
   - Document: `python scripts/build_vectorstore.py --data-dir ./data --output ./chroma_db`

3. **`.github/workflows/lint.yml`** (optional CI)
   - Run `ruff` linting and `pytest` on push to main
   - Does not run agent queries (avoids API cost in CI)

4. **`.gitignore` updates**
   - Add: `.env`, `tutor.db`, `chroma_db/` (if not committing vectorstore), `*.pyc`, `__pycache__/`

5. **`scripts/ingest_manual.py`** — placeholder script
   - Committed to repo with clear TODO comment
   - Accepts `--pdf-path` argument for when satellite manual PDF is provided

---

## Phase 8 — Satellite Operations Manual Integration [PENDING DATA]

**Goal:** Extend the agent's knowledge base to include satellite operations procedures and manuals once that data is acquired.

**Status:** Pipeline is designed and ready. Awaiting PDF source material.

### When data is available:

1. Run `scripts/ingest_manual.py --pdf-path /path/to/satellite_ops_manual.pdf`
   - This will use the same docling extraction + semantic chunking pattern from `NPS_tutor.ipynb` (cells 8–26)
   - Appends new chunks to `data/manual.json`

2. Rebuild vectorstore: `python scripts/build_vectorstore.py`

3. The existing `semantic_retrieve_manual_w_scores` tool in `agent/agent.py` will automatically surface the new content — no agent changes needed

4. Add new eval questions covering satellite ops topics to `data/eval_questions.json`

5. Update system prompt in `agent/agent.py` to mention satellite operations manual as a knowledge source

---

## Evaluation Summary

| Dimension | Method | Storage |
|---|---|---|
| Response correctness | Rule-based scoring in `eval/evaluations.py` (existing) | `eval_scores` table |
| Reasoning quality | Step/equation marker detection (existing) | `eval_scores` table |
| Pedagogical tone | Tutor-language pattern matching (existing) | `eval_scores` table |
| Grounding/citation | Course material reference detection (existing) | `eval_scores` table |
| Tool selection accuracy | Expected vs. actual tool calls per question | `eval_scores` table |
| Retrieval quality | Mean similarity score of top-3 retrieved docs | `tool_usage_logs` table |
| RAG faithfulness | RAGAS `faithfulness` metric (optional, gated by env var) | `eval_scores` table |
| Student progress | Score trends over time per user per topic | Queried from `eval_scores` |

---

## Dependencies Added in This Project

| Package | Purpose |
|---|---|
| `anthropic>=0.75.0` | Claude API client |
| `langchain-anthropic` | LangGraph + Claude tool-calling integration |
| `streamlit>=1.35.0` | Web UI |
| `python-dotenv>=1.0.0` | `.env` file loading |
| `plotly>=5.0.0` | Dashboard charts |
| `ragas>=0.4.1` | Optional RAG evaluation metrics |

All other dependencies (`langchain`, `langgraph`, `chromadb`, `sentence-transformers`, `docling`, etc.) are already in `requirements.txt`.
