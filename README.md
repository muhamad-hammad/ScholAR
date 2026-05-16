---
title: ScholAR Backend
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# ScholAR — Agentic RAG for Research Papers

[![CI](https://github.com/muhamad-hammad/ResearchPaperReaderRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/muhamad-hammad/ResearchPaperReaderRAG/actions/workflows/ci.yml)

**Live demo: https://ask-scholar.vercel.app/**

ScholAR is an agentic Retrieval-Augmented Generation (RAG) system for scientific papers. Upload a PDF, then ask targeted questions or generate a structured summary — answers are grounded in retrieved passages from the paper.

Built on **LangChain + LangGraph** for the agent workflow, **ChromaDB** for vector retrieval, **FastAPI** for the backend, and **Next.js + Tailwind** for the UI.

## Architecture

```
┌──────────────────────┐        ┌──────────────────────────────┐
│  Next.js frontend    │  HTTP  │  FastAPI backend             │
│  (Vercel)            │ ─────► │  (Hugging Face Spaces)       │
│  ask-scholar.vercel  │        │  LangGraph + Chroma + HF     │
└──────────────────────┘        └──────────────────────────────┘
```

- **Frontend** ([web-next/](web-next/)): Next.js 14 app deployed on Vercel. Provides PDF upload, chat, summary, and image-extraction views.
- **Backend** ([api/server.py](api/server.py)): FastAPI service deployed as a Docker Space on Hugging Face. Handles ingestion, retrieval, generation, and figure extraction.
- **Agent workflow** ([src/](src/)): LangGraph router decides between summarization and Q&A paths, with retrieval and grounded generation nodes.
- **LLM providers**: Bring-your-own-key support for OpenAI, Gemini, Groq, OpenRouter, and Grok (xAI). Configured at request time.

## Features

- **PDF ingestion** — token-aware chunking, HF embeddings, in-memory Chroma index per session.
- **Grounded Q&A** — multi-turn chat that retrieves relevant passages before answering.
- **Structured summaries** — motivation, methods, findings, conclusions.
- **Figure extraction** — pulls research figures from the PDF (skips logos/icons via colour-diversity heuristic).
- **Bring-your-own-key** — pick a provider and paste an API key in the UI; nothing is stored server-side.
- **Demo paper** — one-click load of a bundled paper for quick exploration.

## Project Structure

```
api/            — FastAPI backend (server.py)
src/            — Agent core (ingestion, graph nodes, LLM config, workflow)
web-next/       — Next.js 14 frontend (deployed to Vercel)
tests/          — Pytest suite with CI-safe mocks
scripts/        — CLI tools and launch scripts
data/           — Bundled demo PDF
Dockerfile      — Backend image (used by HF Spaces)
```

## Running Locally

### Backend (FastAPI)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

uvicorn api.server:app --port 8000 --reload
```

### Frontend (Next.js)

```powershell
cd web-next
npm install
npm run dev
```

Open http://localhost:3000. The frontend proxies `/api/*` to the local FastAPI server by default.

### Both at once

From `web-next/`:

```powershell
npm run dev:all
```

## Environment Variables

Backend (`.env` at repo root):

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | Default provider (`openai`, `gemini`, `groq`, `openrouter`, `grok`) |
| `LLM_MODEL_ID` | Default model id for the provider |
| `EMBEDDING_MODEL_ID` | HF embedding model (e.g. `sentence-transformers/all-MiniLM-L6-v2`) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Splitter tuning (defaults: 1024 / 128) |
| `RETRIEVER_K` | Top-k chunks to retrieve (default: 4) |
| `LANGSMITH_API_KEY` + `LANGSMITH_TRACING` | Optional LangSmith tracing |
| `OPENAI_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `XAI_API_KEY` | Provider keys (only needed if not passed from the UI) |

Frontend (`web-next/.env.local`):

| Variable | Purpose |
|---|---|
| `NEXT_PUBLIC_BACKEND_URL` | Production backend URL. Leave empty for local dev (uses Next.js proxy). |

## Deployment

- **Backend** → Hugging Face Spaces (Docker SDK). The repo root [Dockerfile](Dockerfile) is the build context; the YAML frontmatter at the top of this README is the Spaces config.
- **Frontend** → Vercel. Set `NEXT_PUBLIC_BACKEND_URL` to the Spaces URL so the browser calls the backend directly (avoids Vercel's proxy timeout on cold starts).

## CLI Usage

```powershell
python scripts/run_single_query.py --pdf data/research_paper.pdf --query "Summarize the paper"
```

## Testing

```powershell
pytest -q
```

CI-only deps (no heavy ML libs):

```powershell
pip install -r requirements-ci.txt
pytest -q
```

## License

Provided for development and educational use. Ensure any models or data you download comply with their respective licenses.
