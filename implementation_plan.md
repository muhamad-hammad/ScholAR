# Implementation Plan — ResearchPaperReaderRAG

## Overview

This document tracks all remaining work for the ResearchPaperReaderRAG project,
organized into five phases from critical bug fixes through developer-experience polish.
Each phase includes a ready-to-use prompt for Claude Code.

---

## Phase 1 — Bug Fixes (Critical) ✅ Complete

**Goal:** Fix three bugs that cause incorrect behavior during normal execution.

### Features
- [x] Fix retriever `k` parameter being read incorrectly from state — fixed in `src/graph_nodes.py:120`, now uses `state.get("k", 4)`
- [x] Fix text splitter ignoring chunk size and overlap values from environment config — fixed in `main.py:68-72`, reads `CHUNK_SIZE`/`CHUNK_OVERLAP` env vars and passes them to `get_text_splitter`
- [x] Fix potential `NameError` in the Streamlit UI when the ingest button is not in scope — resolved; ingest flow was refactored to use `st.session_state.ingested_file` guards, eliminating the bare variable

---

## Phase 2 — Reliability & Session Correctness ✅ Complete

**Goal:** Make the app behave correctly across Streamlit reruns, activate observability,
validate configuration early, and prevent hung or crashing runs.

### Features
- [x] Guard ingestion from re-running on every Streamlit page rerun — `streamlit_app.py:315` checks `st.session_state.ingested_file == uploaded.name`
- [x] Activate LangSmith tracing by setting the required environment variables at startup — `_activate_langsmith()` in `main.py:5`, called at startup in both `main.py` and `streamlit_app.py`
- [x] Validate that the required API key environment variable exists before instantiating the LLM client, with a clear error message if missing — implemented in `src/core_config.py:load_llm()` for all providers (OpenAI, Google, Groq, OpenRouter, Grok/xAI)
- [x] Clean up temporary PDF files after ingestion completes in the Streamlit UI — `streamlit_app.py:333-335` deletes in a `finally` block
- [x] Add a timeout to LLM calls so a hung local model does not block the process indefinitely — thread-pool timeout wired into graph nodes
- [x] Replace the fragile `locals()` check in the CLI script with an explicit boolean flag — no `locals()` call present in `run_single_query.py`

### Remaining work prompt

```
This is a Python RAG application for research papers built with LangChain, LangGraph,
and Streamlit. One reliability item is still missing:

LLM config validation: In `src/llm_adapter.py`, before constructing any LLM client
(OpenAI, Google, Groq, OpenRouter, Grok, HuggingFace), check that the required API key
environment variable for the chosen provider is set and non-empty. Raise a descriptive
ValueError naming the missing variable (e.g. "OPENAI_API_KEY is not set. Export it
before starting the app.") rather than letting the LLM library raise a cryptic internal
error.

After the change, confirm `pytest -q` still passes.
```

---

## Phase 3 — Conversation & Chat History ✅ Complete

**Goal:** Make the application genuinely multi-turn by persisting conversation history
across queries in both the Streamlit UI and the CLI chat loop.

### Features
- [x] Add a `conversation_history` field to the graph state schema — in `main.py`
- [x] Pass prior Q&A turns into the generation node's prompt as context — `graph_nodes.py:217`
- [x] Display conversation history as a chat transcript in the Streamlit UI (user/assistant bubbles) — `streamlit_app.py:444-445`
- [x] Persist conversation history across Streamlit reruns via session state — initialized at `streamlit_app.py:45`, updated at `:473-474`
- [x] Accumulate history in the CLI chat loop and show it on each turn — handled in `run_single_query.py`
- [x] Add a "Clear conversation" button in the Streamlit sidebar — `streamlit_app.py:308`

---

## Phase 4 — Robustness & Quality ✅ Complete

**Goal:** Prevent crashes on large documents, make retrieval quality configurable,
and harden the summarization reduce step against context-window overflow.

### Features
- [x] Add token budget enforcement before the summarization reduce step — `graph_nodes.py:summarization_node`, reads `SUMMARY_MAX_CHARS` (default 12000), keeps last N summaries that fit
- [x] Batch document embedding in chunks to prevent OOM on large PDFs — `ingestion.py:create_vectorstore`, reads `EMBED_BATCH_SIZE` (default 64), logs progress
- [x] Make retriever score threshold configurable via environment variable — `ingestion.py:get_retriever`, reads `RETRIEVER_SCORE_THRESHOLD` (default 0.0)
- [x] Validate that a query and a retriever both exist before entering the graph — `main.py:run_rag_once`, raises `ValueError` with clear message for each missing precondition
- [x] Add a clear error if the retrieved document set is empty — `graph_nodes.py:retrieval_node` sets `final_answer` to a user-friendly message and `_skip_generation=True`; generation node returns early; procedural fallback in `main.py` short-circuits before generation

### Claude Code Prompt

```
This is a Python RAG application for research papers built with LangChain, LangGraph,
and Streamlit. Implement the following robustness improvements:

1. Summarization token budget: In the Map-Reduce summarization node, the reduce step
   concatenates all intermediate chunk summaries and passes them to the LLM in a single
   call. For large papers this can exceed the model's context window. Before the reduce
   call, estimate the total character count of the concatenated summaries and truncate
   to a configurable maximum (default 12000 characters, overridable via a
   SUMMARY_MAX_CHARS env var), keeping the last N summaries that fit rather than the
   first N.

2. Batched embedding: In `src/ingestion.py`, in `create_vectorstore`, embed documents
   in batches of at most 64 at a time (configurable via EMBED_BATCH_SIZE env var)
   rather than passing all chunks at once to Chroma.from_documents. This prevents
   OOM errors on large PDFs. Log progress (e.g., "Embedding batch 3/10...").

3. Retriever score threshold: In `src/ingestion.py`, in `get_retriever`, read a
   RETRIEVER_SCORE_THRESHOLD env var (float, default 0.0, meaning no filter). If the
   value is greater than 0, configure the retriever to discard results below that
   similarity score. Document the env var in .env with a comment.

4. Pre-graph validation: In `main.py`, before invoking the compiled graph (or the
   procedural fallback), check that: (a) the query string is non-empty after stripping
   whitespace, and (b) a retriever object is available. Raise a descriptive ValueError
   for each missing precondition rather than letting the graph fail mid-execution.

5. Empty retrieval handling: In `src/graph_nodes.py`, in the retrieval node, if the
   retriever returns zero documents, set the answer field in state to a user-friendly
   message ("No relevant content found in the document for this query.") and set a
   flag that causes the graph to skip generation/summarization and go directly to END.
   Currently it raises ValueError — change it to route gracefully instead.

After all changes, confirm `pytest -q` still passes.
```

---

## Phase 5 — Developer Experience & CLI Polish ✅ Complete

**Goal:** Make the CLI useful for batch workflows and debugging; add output formatting
and performance visibility.

### Features
- [x] Add `--verbose` flag to CLI that prints retrieved chunks, routing decision, and node timings — `run_single_query.py`
- [x] Add `--output` flag to CLI for JSON output (question, answer, sources, duration) — `run_single_query.py`
- [x] Add `--batch` flag to CLI that accepts a plain-text file of queries (one per line) and writes results to a JSON Lines file — `run_single_query.py`
- [x] Print per-node timing breakdown at the end of each CLI query — `run_single_query.py:107-119`
- [x] Add a `--top-k` CLI argument to override the retriever k value at runtime — `run_single_query.py`

---

## Completion Checklist

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Bug fixes (dict access, splitter args, Streamlit scope) | ✅ Done |
| 2 | Reliability (session guard, tracing, validation, timeout, cleanup) | ✅ Done |
| 3 | Conversation & chat history | ✅ Done |
| 4 | Robustness (token budget, batching, score threshold, validation) | ✅ Done |
| 5 | CLI polish (verbose, JSON output, batch, timing) | ✅ Done |
