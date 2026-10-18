# IoT Department Assistant 🎓

A local **Retrieval-Augmented Generation (RAG)** chatbot that answers student
questions from two kinds of official document: the **IoT B.Tech syllabus**
(all four years) and the **academic calendar**. Every answer is grounded in the
documents and cites its sources (file name and page). If the answer is not in
the documents, the bot says so instead of guessing.

Everything runs **locally** — no cloud account, no API keys. The language model
and embeddings run through [Ollama](https://ollama.com), and the vector store is
a local [ChromaDB](https://www.trychroma.com) database.

---

## What it can answer

| Document | Example questions |
|---|---|
| **Syllabus** (years 1–4) | "What topics are in Operating Systems?" · "List the subjects in 3rd year 1st semester." · "Which lab courses are in 4th year?" · "Credits for 22PC1IN401?" |
| **Academic calendar** | "When do classes commence?" · "When do the semester end exams start?" · "When is the next holiday?" |

Anything else (fees, hostel rules, results) is **out of scope** — the bot
refuses rather than inventing an answer.

---

## Features

- **Grounded answers with citations** — `[file, p.X]` after each statement.
- **Year/semester filters** — set from the sidebar or auto-detected from the
  question ("2nd year", "sem 5", "III year").
- **Hybrid search** — dense vector search + BM25 keyword search fused with
  Reciprocal Rank Fusion, so exact course codes are found.
- **Smart chunking** — one chunk per course (split by unit if long) plus a
  per-semester summary chunk so "list all subjects" works; one chunk per
  calendar event with parsed dates.
- **Query routing** — exam/holiday/date words search the calendar; unit/credit/
  topic words search the syllabus.
- **Calendar awareness** — events are tagged upcoming/passed relative to today.
- **OCR fallback** — scanned calendars are read with Tesseract.
- **Refusal behaviour** — exact not-found reply when the context is insufficient.
- **Evaluation script** — measures retrieval accuracy and refusal rate.

---

## Architecture

```
┌─────────────┐   question    ┌──────────────┐   retrieve    ┌────────────┐
│  Streamlit  │ ────────────▶ │   FastAPI    │ ────────────▶ │  ChromaDB  │
│  frontend   │ ◀──────────── │   backend    │ ◀──────────── │  (vectors) │
└─────────────┘  answer+cites └──────┬───────┘   top chunks  └────────────┘
                                     │ prompt + context
                                     ▼
                               ┌──────────┐
                               │  Ollama  │  LLM + embeddings
                               └──────────┘
```

**Ingestion** (run once, and again when a PDF changes): load PDFs → clean text →
detect year/semester + chunk → embed with Ollama → store in ChromaDB.

**Query** (every question): detect filters + route → hybrid search → build prompt
with today's date → local LLM answers from context only → return answer + sources.

---

## Project structure

```
app/
├── config.py              # all settings from .env
├── embeddings.py          # Ollama embeddings (batched, retried)
├── retrieval.py           # hybrid search, filters, routing, RRF
├── llm.py                 # Ollama chat (blocking + streaming)
├── prompts.py             # grounded system prompt + citations
├── answer.py              # retrieve → threshold → LLM → response
├── api.py                 # FastAPI backend
├── retrieve_cli.py        # test retrieval alone (no LLM)
├── ingestion/
│   ├── loader.py          # PyMuPDF text/tables + OCR fallback
│   ├── cleaner.py         # strip headers/footers/page numbers
│   ├── metadata.py        # year/sem/code/date detection
│   ├── chunkers.py        # syllabus + calendar chunkers
│   ├── pipeline.py        # full ingestion pipeline
│   ├── inspect_pdfs.py    # Stage-2 extraction inspector
│   └── inspect_chunks.py  # Stage-3 chunk inspector
└── vectorstore/
    ├── base.py            # abstract VectorStore interface
    └── chroma_store.py    # ChromaDB implementation
frontend/streamlit_app.py  # chat UI
eval/
├── questions.csv          # test question set (editable)
└── run_eval.py            # metrics + chart
documents/
├── syllabus/              # put syllabus PDFs here
└── academic_calendar/     # put calendar PDFs here
tests/                     # pytest suite (145 tests)
```

---

## Setup

### 1. Prerequisites

- **Python 3.11 or 3.12** (3.13 also works). Tick "Add Python to PATH" on Windows.
- **Ollama** — install from [ollama.com](https://ollama.com), then pull the models:
  ```bash
  ollama pull llama3.2:3b        # chat model (use llama3.2:1b on low-RAM machines)
  ollama pull nomic-embed-text   # embedding model
  ollama list                    # confirm both are listed
  ```
- **Tesseract OCR** — only needed if your calendar PDFs are scanned images.
  - Windows: the [UB Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki)
  - macOS: `brew install tesseract`
  - Linux: `sudo apt install tesseract-ocr`

### 2. Create the environment and install dependencies

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure

Copy the example config and edit if needed:

**Windows:** `copy .env.example .env`
**macOS / Linux:** `cp .env.example .env`

Key settings in `.env`:

| Variable | Default | Notes |
|---|---|---|
| `LLM_MODEL` | `llama3.2:3b` | Use `llama3.2:1b` on machines with little free RAM |
| `EMBED_MODEL` | `nomic-embed-text` | |
| `CHROMA_PATH` | `data/chroma` | Where vectors are stored |
| `TOP_K` | `5` | Chunks retrieved per question |
| `SCORE_THRESHOLD` | `0.3` | Below this → "I couldn't find this" |
| `LLM_NUM_CTX` | `4096` | Context window; lower = less memory per request |
| `TESSERACT_CMD` | Windows install path | Path to the Tesseract binary |

### 4. Add your PDFs

- Syllabus PDF(s) → `documents/syllabus/`
- Academic calendar PDF(s) → `documents/academic_calendar/`

### 5. Ingest the documents

```bash
python -m app.ingestion.pipeline
```

This loads, cleans, chunks, embeds, and stores every PDF. It is **idempotent** —
re-running replaces a file's old chunks. Expect a summary like
`Total chunks in store : 427`.

---

## Running the app

Open **two terminals** (activate the venv in both).

**Terminal 1 — backend:**
```bash
uvicorn app.api:app --reload
```
API at `http://127.0.0.1:8000` · interactive docs at `http://127.0.0.1:8000/docs`.

**Terminal 2 — frontend:**
```bash
streamlit run frontend/streamlit_app.py
```
Chat UI at `http://localhost:8501`.

### Command-line tools

```bash
# Test retrieval only (no LLM) — fast
python -m app.retrieve_cli "What topics are in Operating Systems?"

# Inspect extraction / chunks
python -m app.ingestion.inspect_pdfs
python -m app.ingestion.inspect_chunks
```

### API endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/ask` | `{question, year?, semester?}` → answer + sources |
| `POST` | `/ingest` | Re-run ingestion |
| `GET` | `/health` | Service + Ollama status + chunk count |
| `GET` | `/documents` | Indexed files with chunk counts |

---

## Evaluation

```bash
python -m eval.run_eval             # retrieval metrics (fast)
python -m eval.run_eval --with-llm  # also measures LLM refusal (slower)
```

Reports doc-type accuracy, hit@1/3/5, correct-refusal rate, and average latency;
writes `eval/results.csv` and `eval/results.png`. Edit `eval/questions.csv` to
add your own test questions.

---

## Testing

```bash
pytest            # run all 145 tests
pytest -v         # verbose
```

Tests mock Ollama and use a temporary ChromaDB, so they run without a live model.

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `Connection refused` on port 11434 | Ollama not running | Start the Ollama app or run `ollama serve` |
| `llama-server process has terminated` | Out of RAM | Use `llama3.2:1b`, lower `LLM_NUM_CTX`/`TOP_K`, close other apps, or increase the Windows page file |
| Chunks empty / tiny | Scanned PDF | Install Tesseract and set `TESSERACT_CMD` |
| Course in the wrong year | Heading not detected | Add your heading style to the regexes in `app/ingestion/metadata.py`, then re-ingest |
| "List all subjects" misses some | Summary chunk missing | Confirm ingestion produced `semester_summary` chunks; raise `TOP_K` |
| Course codes not found | Keyword search off | Confirm BM25 + fusion run (they do by default) |
| Very slow answers | Model too big for your RAM | Switch to `llama3.2:1b` |
| Changed the embedding model | Old vectors incompatible | Delete `data/chroma/` and re-ingest |

---

## Tech stack

Python · FastAPI · Streamlit · Ollama (`llama3.2` + `nomic-embed-text`) ·
ChromaDB · rank-bm25 · PyMuPDF · Tesseract · pytest.

## License / use

Use only official documents you are permitted to use. The PDFs under
`documents/` are git-ignored by default.
