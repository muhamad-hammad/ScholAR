# Implementation Plan — ResearchPaperReaderRAG

## Overview

This document tracks all remaining work for the ResearchPaperReaderRAG project,
organized into five phases from critical bug fixes through developer-experience polish.
Each phase includes a ready-to-use prompt for Claude Code.

---

## Phase 1 — Bug Fixes (Critical)

**Goal:** Fix three bugs that cause incorrect behavior during normal execution.

### Features
- [ ] Fix retriever `k` parameter being read incorrectly from state (dict accessed via `getattr` instead of `.get`)
- [ ] Fix text splitter ignoring chunk size and overlap values from environment config
- [ ] Fix potential `NameError` in the Streamlit UI when the ingest button is not in scope

### Claude Code Prompt

```
This is a Python RAG (Retrieval-Augmented Generation) application for research papers
built with LangChain, LangGraph, and Streamlit. There are three bugs to fix:

1. In the LangGraph node responsible for document retrieval, the top-k value is being
   read from the graph state using `getattr()`. The state object is a plain Python dict
   (TypedDict at type-check time, dict at runtime), so attribute access will never find
   the key. Replace the `getattr(state, ...)` call with the correct dict `.get(key, default)`
   pattern wherever top-k is read from state.

2. In the ingestion module, the token-aware text splitter is instantiated without passing
   chunk size or chunk overlap arguments. These values should be read from environment
   variables (CHUNK_SIZE and CHUNK_OVERLAP). Fix the instantiation so those env vars
   are respected, and add sensible integer defaults if the vars are absent.

3. In the Streamlit app, a variable that tracks whether the ingest button was clicked is
   used in a conditional check before it is guaranteed to be defined. Initialize it to
   False at the top of the relevant scope so a NameError cannot occur on any code path.

After fixing, confirm that `pytest -q` still passes.
```

---

## Phase 2 — Reliability & Session Correctness

**Goal:** Make the app behave correctly across Streamlit reruns, activate observability,
validate configuration early, and prevent hung or crashing runs.

### Features
- [ ] Guard ingestion from re-running on every Streamlit page rerun (use session state flag)
- [ ] Activate LangSmith tracing by setting the required environment variables at startup
- [ ] Validate that the required API key environment variable exists before instantiating the LLM client, with a clear error message if missing
- [ ] Clean up temporary PDF files after ingestion completes in the Streamlit UI
- [ ] Add a timeout to LLM calls so a hung local model does not block the process indefinitely
- [ ] Replace the fragile `locals()` check in the CLI script with an explicit boolean flag

### Claude Code Prompt

```
This is a Python RAG application for research papers built with LangChain, LangGraph,
and Streamlit. Implement the following reliability improvements:

1. Streamlit reruns: Streamlit re-executes the entire script on every user interaction.
   The ingestion step (PDF loading, chunking, vectorstore creation) is expensive and must
   only run once per uploaded file. Use Streamlit session state to store a flag and the
   resulting retriever so that repeated reruns skip ingestion if it has already succeeded
   for the current upload.

2. LangSmith tracing: The project has LANGSMITH_API_KEY and LANGSMITH_TRACING in its
   .env file, but tracing is never activated. At application startup (before any LangChain
   calls), read those environment variables and set the corresponding os.environ keys that
   LangSmith requires (LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY). Only activate if both
   values are present and non-empty.

3. LLM config validation: Before constructing any LLM client (OpenAI, Google, HuggingFace),
   check that the required API key environment variable for the chosen provider is set and
   non-empty. Raise a descriptive ValueError naming the missing variable rather than letting
   the LLM library raise a cryptic internal error.

4. Temp file cleanup: When the Streamlit app saves an uploaded PDF to a temporary file for
   ingestion, that file is never removed. After ingestion completes (whether it succeeds or
   fails), delete the temporary file in a finally block.

5. LLM call timeout: Wrap LLM inference calls in a timeout so that a hung local model
   (HuggingFace pipeline) does not block the process. Use Python's concurrent.futures with
   a configurable timeout (default 120 seconds, overridable via an LLM_TIMEOUT_SECONDS env
   var). On timeout, raise a clear RuntimeError.

6. CLI variable tracking: In the single-query CLI script, a `locals()` dict lookup is used
   to check whether the LLM was successfully loaded. Replace this with an explicit boolean
   flag variable (e.g., `llm_loaded = False`, set to True after successful load) for
   reliable conditional branching.

After all changes, confirm `pytest -q` still passes.
```

---

## Phase 3 — Conversation & Chat History

**Goal:** Make the application genuinely multi-turn by persisting conversation history
across queries in both the Streamlit UI and the CLI chat loop.

### Features
- [ ] Add a `conversation_history` field to the graph state schema
- [ ] Pass prior Q&A turns into the generation node's prompt as context
- [ ] Display conversation history as a chat transcript in the Streamlit UI (user/assistant bubbles)
- [ ] Persist conversation history across Streamlit reruns via session state
- [ ] Accumulate history in the CLI chat loop and show it on each turn
- [ ] Add a "Clear conversation" button in the Streamlit sidebar

### Claude Code Prompt

```
This is a Python RAG application for research papers built with LangChain, LangGraph,
and Streamlit. Currently every query is treated as independent — there is no memory of
previous turns. Implement multi-turn conversation history:

1. State schema: Add a `conversation_history` field to the graph state TypedDict. It should
   hold a list of dicts, each with "role" ("user" or "assistant") and "content" (str) keys.
   Default to an empty list.

2. Generation node: In the node that produces the final answer, prepend the conversation
   history to the prompt so the LLM has context from prior turns. Format each turn as
   "User: <question>\nAssistant: <answer>" separated by blank lines, placed before the
   current question. Keep history to the last 6 turns maximum to avoid exceeding context
   limits.

3. History accumulation: After each successful query (in both the compiled-graph path and
   the procedural fallback path), append the user query and the assistant answer to
   conversation_history in the state, then pass the updated state into the next query
   invocation.

4. Streamlit UI: Store conversation_history in st.session_state. Render the conversation
   as a chat transcript above the query input using st.chat_message("user") and
   st.chat_message("assistant") bubbles. Add a "Clear conversation" button in the sidebar
   that resets the history.

5. CLI chat loop: In the interactive terminal loop, maintain conversation_history as a
   local list across iterations. Print the full conversation so far at the start of each
   loop iteration (last 6 turns only).

After all changes, confirm `pytest -q` still passes and manually verify that a follow-up
question like "Can you elaborate on that?" receives a contextually aware answer.
```

---

## Phase 4 — Robustness & Quality

**Goal:** Prevent crashes on large documents, make retrieval quality configurable,
and harden the summarization reduce step against context-window overflow.

### Features
- [ ] Add token budget enforcement before the summarization reduce step
- [ ] Batch document embedding in chunks to prevent OOM on large PDFs
- [ ] Make retriever score threshold configurable via environment variable
- [ ] Validate that a query and a retriever both exist before entering the graph
- [ ] Add a clear error if the retrieved document set is empty (no relevant content found)

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

2. Batched embedding: In the ingestion module, when creating the vectorstore from
   document chunks, embed documents in batches of at most 64 at a time (configurable
   via EMBED_BATCH_SIZE env var) rather than passing all chunks at once. This prevents
   OOM errors on large PDFs. Log progress (e.g., "Embedding batch 3/10...").

3. Retriever score threshold: When constructing the retriever from the vectorstore, read
   a RETRIEVER_SCORE_THRESHOLD env var (float, default 0.0, meaning no filter). If the
   value is greater than 0, configure the retriever to discard results below that
   similarity score. Document the env var in the .env file with a comment.

4. Pre-graph validation: Before invoking the compiled graph (or the procedural fallback),
   check that: (a) the query string is non-empty after stripping whitespace, and (b) a
   retriever object is available. Raise a descriptive ValueError for each missing
   precondition rather than letting the graph fail mid-execution.

5. Empty retrieval handling: In the retrieval node, if the retriever returns zero
   documents, set the answer field in state to a user-friendly message ("No relevant
   content found in the document for this query.") and set a flag that causes the graph
   to skip the generation/summarization node and go directly to END.

After all changes, confirm `pytest -q` still passes.
```

---

## Phase 5 — Developer Experience & CLI Polish

**Goal:** Make the CLI useful for batch workflows and debugging; add output formatting
and performance visibility.

### Features
- [ ] Add `--verbose` flag to CLI that prints retrieved chunks, routing decision, and node timings
- [ ] Add `--output` flag to CLI for JSON output (question, answer, sources, duration)
- [ ] Add `--batch` flag to CLI that accepts a plain-text file of queries (one per line) and writes results to a JSON Lines file
- [ ] Print per-node timing breakdown at the end of each CLI query
- [ ] Add a `--top-k` CLI argument to override the retriever k value at runtime

### Claude Code Prompt

```
This is a Python RAG application for research papers. The CLI script accepts a PDF and
a single query and prints an answer. Extend it with the following features:

1. --verbose flag: When passed, print the following after each query:
   - The routing decision (SUMMARY or QNA) and confidence/keyword match
   - The retrieved document chunks (source metadata + first 200 characters of content)
   - Time taken by each graph node in milliseconds

2. --output flag: Accepts a file path. When provided, write the result as a JSON object
   to that file with the following fields: "query", "answer", "sources" (list of unique
   source metadata dicts), "routing" (SUMMARY or QNA), "duration_ms" (total wall time).

3. --batch flag: Accepts a path to a plain-text file containing one query per line.
   When used with --pdf, run all queries against the same ingested vectorstore (ingest
   once, query N times). Write results to a JSON Lines file at the path given by
   --output (required when --batch is used). Print a progress indicator (e.g.,
   "Query 3/10...") to stderr.

4. --top-k flag: Integer argument (default 4) that overrides the number of documents
   the retriever fetches. Pass this value through to the graph state so the retrieval
   node respects it.

5. Timing breakdown: After every query (batch or single), print a summary table to
   stderr showing wall-clock time in ms for: ingestion (if run), routing, retrieval,
   generation/summarization, total.

Ensure all new flags degrade gracefully (e.g., --batch without --output prints a clear
usage error). After all changes, confirm `pytest -q` still passes.
```

---

## Completion Checklist

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Bug fixes (dict access, splitter args, Streamlit scope) | [ ] |
| 2 | Reliability (session guard, tracing, validation, timeout, cleanup) | [ ] |
| 3 | Conversation & chat history | [ ] |
| 4 | Robustness (token budget, batching, score threshold, validation) | [ ] |
| 5 | CLI polish (verbose, JSON output, batch, timing) | [ ] |
