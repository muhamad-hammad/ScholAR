import concurrent.futures
import os
from typing import Any
from src.core_config import ResearchRAGState
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable


def _call_llm(llm: Any, prompt: str) -> str:
    timeout = int(os.getenv("LLM_TIMEOUT_SECONDS", "120"))

    def _invoke():
        try:
            if callable(llm):
                res = llm(prompt)
            elif hasattr(llm, "generate"):
                res = llm.generate(prompt)
            else:
                res = llm.pipeline(prompt)
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}") from e

        if isinstance(res, str):
            return res
        if isinstance(res, dict):
            return res.get("generated_text") or str(res)
        if isinstance(res, list):
            first = res[0]
            if isinstance(first, dict):
                return first.get("generated_text") or str(first)
            return str(first)
        return str(res)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_invoke)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise RuntimeError(f"LLM call timed out after {timeout} seconds")


def router_node(state: ResearchRAGState) -> ResearchRAGState:
    """
    Router node that classifies `state['user_query']` into an intent label.

    Detailed responsibilities (comment-only):
    - Use a small, efficient classifier chain or a single-shot LLM call to classify
      a user query into one of the supported intents: 'SUMMARY' or 'QNA'.
    - Update `state['query_intent']` with the classification result.
    - Optionally add reasoning metadata to the state for traceability (e.g., why
      the query was labeled SUMMARY).
    - Keep the classifier lightweight to avoid unnecessary compute when many queries are quick metadata checks.
    """
    # Simple rule-based router to keep this lightweight and testable.
    query = state.get("user_query", "")
    if not isinstance(query, str):
        query = str(query)

    q = query.strip().lower()
    summary_keywords = [
        "summarize",
        "summary",
        "abstract",
        "overview",
        "explain",
        "describe",
        "key findings",
        "conclusions",
    ]

    intent = "QNA"
    reason = "default"
    for kw in summary_keywords:
        if kw in q:
            intent = "SUMMARY"
            reason = f"matched_keyword:{kw}"
            break

    # Update state with the chosen intent and optional trace metadata
    state["query_intent"] = intent
    meta = state.get("meta", {})
    meta.setdefault("router", {})
    meta["router"]["reason"] = reason
    state["meta"] = meta
    return state


def retrieval_node(state: ResearchRAGState) -> ResearchRAGState:
    """
    Retrieval node executed only when intent == 'QNA'.

    Detailed responsibilities (comment-only):
    - Use the configured Retriever (from Chroma) to fetch the top-k most relevant
      document chunks for `state['user_query']`.
    - Store the retrieved LangChain `Document` objects in `state['retrieved_docs']`.
    - Attach retrieval metadata such as scores, provenance (source file, page),
      and any highlights used for debugging.
    """
    query = state.get("user_query")
    if not query:
        raise ValueError("retrieval_node requires state['user_query'] to be set")

    # Expect a Retriever object to be provided in state under 'retriever'
    retriever = state.get("retriever")
    if retriever is None:
        raise ValueError("retrieval_node requires a 'retriever' in state")

    # Try common Retriever APIs in order of preference
    docs = None
    try:
        docs = retriever.invoke(query)
    except Exception:
        try:
            docs = retriever.get_relevant_documents(query)
        except Exception:
            try:
                docs = retriever.retrieve(query)
            except Exception:
                try:
                    docs = retriever.similarity_search(query, k=state.get("k", 4))
                except Exception as e:
                    raise RuntimeError("Retriever could not execute a search method") from e

    # Ensure we have a list
    if docs is None:
        docs = []

    # Attach provenance metadata for each document where possible
    retrieval_meta = []
    for d in docs:
        md = getattr(d, "metadata", {}) or {}
        entry = {
            "source": md.get("source"),
            # Some vectorstores embed scores in metadata under 'score' or similar.
            "score": md.get("score") if isinstance(md.get("score"), (int, float)) else None,
        }
        retrieval_meta.append(entry)

    state["retrieved_docs"] = docs
    state["retrieval_metadata"] = retrieval_meta
    return state


def summarization_node(state: ResearchRAGState, llm: Any) -> ResearchRAGState:
    """
    Summarization node executed when intent == 'SUMMARY'.

    Detailed responsibilities (comment-only):
    - Implement a Map-Reduce (or Refine) summarization pattern to handle arbitrarily
      long research papers without exceeding the LLM context window.
      * Map step: iterate over raw document chunks, produce intermediate summaries
        and append them to `state['raw_summary_parts']`.
      * Reduce step: synthesize the intermediate summaries into a single cohesive abstract
        using a final LLM call.
    - Ensure the reduce prompt enforces an "abstract-style" output and includes
      explicit instructions to preserve key claims, methods, and results.
    - Save the final text into `state['final_answer']`.
    - Record any intermediate artifacts for LangSmith traces (e.g., partial summaries).
    """
    # Map-Reduce summarization implementation.
    # Map step: create short summaries for each document/chunk.
    docs = state.get("retrieved_docs") or state.get("documents") or []
    if not docs:
        raise ValueError("summarization_node requires 'retrieved_docs' or 'documents' in state")

    intermediate = []
    for d in docs:
        text = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
        prompt = (
            "Summarize the following document chunk in 2-3 sentences focusing on main claims, "
            "methods, and results. Be concise and factual.\n\n" + text + "\n\nSummary:"
        )
        summary = _call_llm(llm, prompt).strip()
        intermediate.append(summary)

    # Save intermediate summaries
    state["raw_summary_parts"] = intermediate

    # Reduce step: synthesize intermediate summaries into a cohesive abstract
    reduce_prompt = (
        "You are given multiple short summaries of parts of a scientific paper. "
        "Synthesize them into a single concise abstract (150-300 words) that preserves key claims, methods, and results. "
        "Write in a neutral academic tone.\n\n"
        + "\n\n".join(intermediate)
        + "\n\nAbstract:"
    )
    final = _call_llm(llm, reduce_prompt).strip()
    state["final_answer"] = final

    # Trace metadata
    meta = state.get("meta", {})
    meta.setdefault("summarization", {})
    meta["summarization"]["parts"] = len(intermediate)
    state["meta"] = meta
    return state


def generation_node(state: ResearchRAGState, llm: Any) -> ResearchRAGState:
    """
    Generation node for Q&A path executed after retrieval_node.

    Detailed responsibilities (comment-only):
    - Construct a grounded prompt that concatenates the top retrieved chunks and
      the `state['user_query']` while explicitly instructing the LLM to only answer
      from the provided context (to minimize hallucinations).
    - Prepend up to the last 6 Q&A turns from conversation_history so the LLM has
      context for follow-up questions.
    - Include provenance reporting in the output (e.g., "Answer based on section X, page Y").
    - Store the final generated answer in `state['final_answer']`.
    """
    # Grounded generation using retrieved document chunks.
    query = state.get("user_query")
    if not query:
        raise ValueError("generation_node requires state['user_query']")

    docs = state.get("retrieved_docs") or []
    history = state.get("conversation_history") or []

    # Build context by concatenating top-k documents with provenance markers
    context_parts = []
    for idx, d in enumerate(docs[:8]):
        md = getattr(d, "metadata", {}) or {}
        src = md.get("source") or md.get("file") or f"doc_{idx}"
        content = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
        context_parts.append(f"[Source: {src}]\n{content}")

    context = "\n\n---\n\n".join(context_parts)

    # Format the last 6 Q&A pairs (12 entries) from conversation history
    history_section = ""
    if history:
        recent = history[-12:]
        formatted = []
        for turn in recent:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        if formatted:
            history_section = "Previous conversation:\n" + "\n\n".join(formatted) + "\n\n"

    prompt = (
        "You are an expert assistant. Answer the user's question ONLY using the information in the provided context. "
        "If the answer cannot be found in the context, say 'I don't know' or state that the information is not available. "
        "Provide brief provenance for your answers (which source/section).\n\n"
        + history_section
        + "Context:\n"
        + context
        + "\n\nUser question:\n"
        + query
        + "\n\nAnswer (be concise, include provenance):"
    )

    answer = _call_llm(llm, prompt).strip()
    state["final_answer"] = answer

    # Attach provenance info if available
    meta = state.get("meta", {})
    meta.setdefault("generation", {})
    meta["generation"]["sources_used"] = [getattr(d, "metadata", {}).get("source") for d in docs[:8]]
    state["meta"] = meta
    return state


def determine_next_node(state: ResearchRAGState) -> str:
    """
    Conditional function to direct the graph's control flow from the router node.

    Implementation contract (comment-only):
    - Inspect `state['query_intent']` and return the string name of the next node:
      either 'retrieval_node' or 'summarization_node'.
    - This function must be deterministic and well-instrumented for tracing.
    """
    # Read intent from state, normalize to uppercase for robustness
    intent = None
    if isinstance(state, dict):
        intent = state.get("query_intent")
    # If not explicitly set, attempt a lightweight fallback from user_query
    if not intent:
        uq = (state.get("user_query") or "")
        if isinstance(uq, str) and "?" in uq:
            intent = "QNA"
        else:
            # Default to SUMMARY for short 'summarize' style queries, else QNA
            if isinstance(uq, str) and any(k in uq.lower() for k in ["summarize", "summary", "abstract"]):
                intent = "SUMMARY"
            else:
                intent = "QNA"

    intent = (intent or "QNA").upper()

    # Map the intent to the node name expected by the StateGraph
    if intent == "SUMMARY":
        return "summarization_node"
    # Default and 'QNA' go to retrieval
    return "retrieval_node"
