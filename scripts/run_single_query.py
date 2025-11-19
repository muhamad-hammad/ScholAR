"""CLI helper to run a single query against the RAG system.

Usage examples:
  python scripts/run_single_query.py --pdf paper.pdf --query "Summarize"
  python scripts/run_single_query.py --query "What is the main contribution?"
"""
import os
import argparse
import sys

from src.llm_adapter import adapt_llm
from src.workflow_builder import build_research_rag_graph
from main import run_ingestion_pipeline, run_rag_once
from src.core_config import load_hf_pipeline


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", help="Path to PDF to ingest (optional)")
    parser.add_argument("--query", required=True, help="The user query to run")
    parser.add_argument("--llm", help="Optional HF LLM model id to load locally")
    args = parser.parse_args(argv)

    retriever = None
    if args.pdf:
        # set env var used by run_ingestion_pipeline
        os.environ["PDF_INPUT_PATH"] = args.pdf
        retriever = run_ingestion_pipeline()

    llm = None
    if args.llm:
        try:
            llm_raw = load_hf_pipeline(args.llm)
            llm = adapt_llm(llm_raw)
        except Exception as e:
            print("Failed to load LLM locally; proceeding without it:", e, file=sys.stderr)

    compiled = None
    if llm is not None and retriever is not None:
        try:
            compiled = build_research_rag_graph(llm=llm, retriever=retriever)
        except Exception:
            compiled = None

    # Ensure adapter is applied to llm if provided as raw callable
    if llm is None and 'llm_raw' in locals():
        llm = adapt_llm(locals().get('llm_raw'))

    answer, state = run_rag_once(compiled, args.query, retriever=retriever, llm=llm)
    print("=== ANSWER ===")
    print(answer)
    print("=== STATE ===")
    print(state)


if __name__ == "__main__":
    main()
