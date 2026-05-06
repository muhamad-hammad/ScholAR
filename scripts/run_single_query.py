"""CLI helper to run queries against the ResearchPaperRAG system.

Usage examples:
  python scripts/run_single_query.py --pdf paper.pdf --query "Summarize"
  python scripts/run_single_query.py --pdf paper.pdf --query "What is the main finding?" --verbose
  python scripts/run_single_query.py --pdf paper.pdf --query "Methods?" --output result.json
  python scripts/run_single_query.py --pdf paper.pdf --batch queries.txt --output results.jsonl
  python scripts/run_single_query.py --pdf paper.pdf --query "Methods?" --top-k 8
"""
import json
import os
import sys
import time
import argparse

from dotenv import load_dotenv

from src.llm_adapter import adapt_llm
from src.core_config import load_hf_pipeline
from src.graph_nodes import (
    router_node,
    retrieval_node,
    generation_node,
    summarization_node,
    determine_next_node,
)
from main import run_ingestion_pipeline, _activate_langsmith


def _ms(elapsed: float) -> int:
    return int(elapsed * 1000)


def run_query_timed(query: str, retriever, llm, top_k: int = 4) -> tuple:
    """Run a single RAG query through the procedural path and return (answer, state, timings).

    Runs each node individually so wall-clock time per node can be measured.
    timings keys: routing, retrieval, generation, summarization, total
    """
    timings: dict = {}
    state: dict = {
        "user_query": query,
        "meta": {},
        "conversation_history": [],
        "k": top_k,
        "retriever": retriever,
    }

    t_start = time.perf_counter()

    t0 = time.perf_counter()
    state = router_node(state)
    timings["routing"] = _ms(time.perf_counter() - t0)

    t0 = time.perf_counter()
    state = retrieval_node(state)
    timings["retrieval"] = _ms(time.perf_counter() - t0)

    next_node = determine_next_node(state)

    effective_llm = llm
    if effective_llm is None:
        if next_node == "summarization_node":
            def effective_llm(p: str) -> str:
                return (p or "")[:200]
        else:
            def effective_llm(p: str) -> str:
                return "I don't know"

    t0 = time.perf_counter()
    if next_node == "summarization_node":
        state = summarization_node(state, effective_llm)
        timings["summarization"] = _ms(time.perf_counter() - t0)
        timings["generation"] = 0
    else:
        state = generation_node(state, effective_llm)
        timings["generation"] = _ms(time.perf_counter() - t0)
        timings["summarization"] = 0

    timings["total"] = _ms(time.perf_counter() - t_start)
    return state.get("final_answer"), state, timings


def _unique_sources(state: dict) -> list:
    docs = state.get("retrieved_docs") or []
    seen: set = set()
    sources = []
    for d in docs:
        md = getattr(d, "metadata", {}) or {}
        key = str(md)
        if key not in seen:
            sources.append(md)
            seen.add(key)
    return sources


def _build_result(query: str, answer: str, state: dict, timings: dict) -> dict:
    return {
        "query": query,
        "answer": answer,
        "sources": _unique_sources(state),
        "routing": state.get("query_intent", "QNA"),
        "duration_ms": timings.get("total", 0),
    }


def _print_timing_table(timings: dict, ingestion_ms: int | None, file=sys.stderr) -> None:
    print("\n--- Timing breakdown ---", file=file)
    if ingestion_ms is not None:
        print(f"  Ingestion:         {ingestion_ms:>6} ms", file=file)
    print(f"  Routing:           {timings.get('routing', 0):>6} ms", file=file)
    print(f"  Retrieval:         {timings.get('retrieval', 0):>6} ms", file=file)
    gen_ms = timings.get("generation", 0)
    sum_ms = timings.get("summarization", 0)
    if sum_ms:
        print(f"  Summarization:     {sum_ms:>6} ms", file=file)
    else:
        print(f"  Generation:        {gen_ms:>6} ms", file=file)
    print(f"  Total:             {timings.get('total', 0):>6} ms", file=file)
    print("------------------------", file=file)


def _print_verbose(state: dict, timings: dict, file=sys.stderr) -> None:
    meta = state.get("meta", {})
    intent = state.get("query_intent", "QNA")
    reason = meta.get("router", {}).get("reason", "")
    print(f"\n[VERBOSE] Routing: {intent}  reason={reason}", file=file)

    docs = state.get("retrieved_docs") or []
    print(f"[VERBOSE] Retrieved {len(docs)} chunk(s):", file=file)
    for i, d in enumerate(docs):
        md = getattr(d, "metadata", {}) or {}
        content = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
        print(f"  [{i + 1}] source={md.get('source', 'unknown')} | {content[:200]!r}", file=file)

    print("[VERBOSE] Node timings (ms):", file=file)
    for key in ("routing", "retrieval", "generation", "summarization"):
        ms = timings.get(key, 0)
        if ms:
            print(f"  {key}: {ms} ms", file=file)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Run queries against the ResearchPaperRAG system."
    )
    parser.add_argument("--pdf", help="Path to PDF to ingest")
    parser.add_argument("--query", help="Single query (mutually exclusive with --batch)")
    parser.add_argument("--llm", help="HF model id for local LLM (optional)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print routing decision, retrieved chunks, and per-node timings")
    parser.add_argument("--output", metavar="FILE",
                        help="Output file: JSON for single query, JSON Lines for --batch")
    parser.add_argument("--batch", metavar="FILE",
                        help="Plain-text file of queries (one per line); requires --output")
    parser.add_argument("--top-k", type=int, default=4, dest="top_k",
                        help="Number of documents the retriever fetches (default: 4)")
    args = parser.parse_args(argv)

    # Validate flag combinations
    if args.batch and not args.output:
        parser.error("--batch requires --output to specify the results file")
    if args.query and args.batch:
        parser.error("--query and --batch are mutually exclusive")
    if not args.query and not args.batch:
        parser.error("Provide --query for a single query or --batch FILE for batch mode")

    load_dotenv()
    _activate_langsmith()

    # Ingestion
    ingestion_ms: int | None = None
    retriever = None
    if args.pdf:
        os.environ["PDF_INPUT_PATH"] = args.pdf
        os.environ["RETRIEVER_K"] = str(args.top_k)
        t0 = time.perf_counter()
        retriever = run_ingestion_pipeline()
        ingestion_ms = _ms(time.perf_counter() - t0)
        print(f"Ingestion complete ({ingestion_ms} ms)", file=sys.stderr)

    # LLM loading
    llm = None
    llm_raw = None
    if args.llm:
        try:
            llm_raw = load_hf_pipeline(args.llm)
            llm = adapt_llm(llm_raw)
        except Exception as exc:
            print(f"Failed to load LLM; proceeding without it: {exc}", file=sys.stderr)
    if llm is None and llm_raw is not None:
        llm = adapt_llm(llm_raw)

    def _run_one(query: str, show_ingestion_ms: int | None = None) -> tuple:
        answer, state, timings = run_query_timed(query, retriever, llm, top_k=args.top_k)
        if args.verbose:
            _print_verbose(state, timings)
        _print_timing_table(timings, show_ingestion_ms)
        return answer, state, timings

    # Batch mode
    if args.batch:
        with open(args.batch, encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]
        total = len(queries)
        with open(args.output, "w", encoding="utf-8") as out_f:
            for idx, query in enumerate(queries, 1):
                print(f"Query {idx}/{total}...", file=sys.stderr)
                show_ms = ingestion_ms if idx == 1 else None
                answer, state, timings = _run_one(query, show_ingestion_ms=show_ms)
                out_f.write(json.dumps(_build_result(query, answer, state, timings)) + "\n")
        print(f"Batch complete — {total} queries written to {args.output}", file=sys.stderr)
        return

    # Single-query mode
    answer, state, timings = _run_one(args.query, show_ingestion_ms=ingestion_ms)
    print("=== ANSWER ===")
    print(answer)
    if args.output:
        result = _build_result(args.query, answer, state, timings)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Result written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
