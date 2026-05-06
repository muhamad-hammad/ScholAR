import os
from dotenv import load_dotenv


def _activate_langsmith() -> None:
    """Activate LangSmith tracing if both LANGSMITH_API_KEY and LANGSMITH_TRACING are set."""
    api_key = os.getenv("LANGSMITH_API_KEY", "")
    tracing = os.getenv("LANGSMITH_TRACING", "")
    if api_key and tracing:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = api_key

# Imports for LangChain / LangGraph components and local modules
from langchain_core.documents import Document
from src.core_config import load_hf_pipeline, load_hf_embeddings, load_llm, ResearchRAGState
from src.ingestion import load_documents, get_text_splitter, create_vectorstore, get_retriever
from src.workflow_builder import build_research_rag_graph
from src.graph_nodes import (
    router_node,
    retrieval_node,
    generation_node,
    summarization_node,
    determine_next_node,
)

# Application entry point for the Agentic Research RAG system.
# This file contains the high-level orchestration signatures and docstring-level comments
# explaining how the runtime should be wired. No implementation logic is included here.


def run_ingestion_pipeline() -> object:
    """
    Ingestion pipeline entry-point.

    Responsibilities (comment-only):
    - Load environment configuration (paths, model IDs, tokenizer name).
    - Initialize a specialized document loader designed for research PDFs (Dedoc or Unstructured).
    - Read the raw research paper found at PDF_INPUT_PATH and convert into a list of
      LangChain `Document` objects while preserving structure (sections, tables, metadata).
    - Instantiate a token-aware `TokenTextSplitter` using the Hugging Face tokenizer
      for the configured LLM model; ensure the splitter uses `from_huggingface_tokenizer`
      so chunk sizes are strictly enforced by token count.
    - Apply the splitter to documents to produce context-preserving chunks.
    - Load the open-source Hugging Face embedding model (EMBEDDING_MODEL_ID) and
      transform chunks into embeddings.
    - Create or persist a ChromaDB vectorstore from the embeddings and return
      a LangChain `Retriever` object configured with the tuned `k` parameter.

    Returns:
        retriever: A configured LangChain Retriever backed by ChromaDB
    """
    # Load environment variables from .env if present
    load_dotenv()

    pdf_path = os.getenv("PDF_INPUT_PATH")
    if not pdf_path:
        raise ValueError("PDF_INPUT_PATH must be set in the environment or passed to run_ingestion_pipeline")

    # Resolve tokenizer and embedding model ids from env
    tokenizer_name = os.getenv("TOKENIZER_NAME") or os.getenv("LLM_MODEL_ID") or os.getenv("EMBEDDING_MODEL_ID")
    embedding_model_id = os.getenv("EMBEDDING_MODEL_ID")

    # Optional persistence directory for Chroma
    persist_dir = os.getenv("CHROMA_PERSIST_DIR")

    # Chunking params (allow overrides via env)
    try:
        chunk_size = int(os.getenv("CHUNK_SIZE", "1024"))
    except Exception:
        chunk_size = 1024
    try:
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "128"))
    except Exception:
        chunk_overlap = 128

    # Retriever param
    try:
        k = int(os.getenv("RETRIEVER_K", "4"))
    except Exception:
        k = 4

    # 1) Load raw documents
    docs = load_documents(pdf_path)

    # 2) Create a token-aware splitter and split documents into chunks
    splitter = get_text_splitter(tokenizer_name or "gpt2", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    try:
        chunks = splitter.split_documents(docs)
    except Exception:
        # Fallback: split_text per document and wrap into Document objects
        chunks = []
        for d in docs:
            text = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
            parts = splitter.split_text(text)
            for p in parts:
                chunks.append(Document(page_content=p, metadata=getattr(d, "metadata", {})))

    # 3) Load embeddings
    embeddings = load_hf_embeddings(embedding_model_id)

    # 4) Create or load vectorstore
    vectordb = create_vectorstore(chunks, embeddings, persist_directory=persist_dir)

    # 5) Return a Retriever tuned with k
    retriever = get_retriever(vectordb, k=k)
    return retriever


def _append_history(state: dict, user_query: str, answer: str) -> None:
    """Append the current Q&A turn to conversation_history in state (mutates state)."""
    if answer is not None:
        hist = state.setdefault("conversation_history", [])
        hist.append({"role": "user", "content": user_query})
        hist.append({"role": "assistant", "content": answer})


def run_rag_chat_loop(compiled_graph: object, retriever: object = None, llm: object = None) -> None:
    """
    Interactive chat loop for running LangGraph compiled workflows.

    Responsibilities (comment-only):
    - Accept user input queries in a loop (CLI or lightweight web UI wrapper).
    - For each incoming query, seed the `ResearchRAGState` with the `user_query`
      and any optionally persisted `conversation_history`.
    - Execute the compiled LangGraph workflow and block until a terminal state is
      returned. The graph execution should be fully traced (LangSmith) if
      environment tracing variables are enabled.
    - Print the conversation history (last 6 Q&A pairs) at the start of each iteration.
    - Accumulate conversation history across turns for multi-turn context.

    Args:
        compiled_graph: The compiled/ready-to-run LangGraph instance.
        retriever: Optional retriever for the procedural fallback path.
        llm: Optional LLM callable for the procedural fallback path.
    """
    print("Entering RAG chat loop. Type 'exit' or Ctrl-C to quit.")
    conversation_history: list = []
    try:
        while True:
            # Print recent history at the top of each turn
            if conversation_history:
                print("\n--- Recent conversation (last 6 turns) ---")
                for turn in conversation_history[-12:]:
                    role = turn.get("role", "").capitalize()
                    content = turn.get("content", "")
                    print(f"{role}: {content}")
                print("---\n")

            user_query = input("User> ").strip()
            if not user_query:
                continue
            if user_query.lower() in ("exit", "quit"):
                print("Exiting.")
                break

            try:
                answer, state = run_rag_once(
                    compiled_graph, user_query,
                    retriever=retriever, llm=llm,
                    conversation_history=conversation_history,
                )
                print("Assistant:", answer)
                conversation_history = state.get("conversation_history", conversation_history)
            except Exception as e:
                print("Error while processing query:", e)
    except KeyboardInterrupt:
        print("Interrupted; exiting chat loop.")


def run_rag_once(
    compiled_graph: object,
    user_query: str,
    retriever: object = None,
    llm: object = None,
    conversation_history: list = None,
):
    """
    Run a single RAG execution and return (final_answer, state).

    Behavior:
    - If `compiled_graph` is a LangGraph compiled object exposing a run/execute/call
      entrypoint, attempt to run it with an initial `ResearchRAGState`.
    - If `compiled_graph` is None or not executable, fall back to a simple
      procedural execution using the pure functions in `src.graph_nodes`.
    - conversation_history is threaded through state so generation_node can use
      prior turns; the completed turn is appended before returning.

    Returns:
        (final_answer: str, state: ResearchRAGState)
    """
    history = list(conversation_history or [])

    # Build initial state, seeding conversation history for the generation node
    state = {"user_query": user_query, "meta": {}, "conversation_history": history}

    # Attach retriever if provided
    if retriever is not None:
        state["retriever"] = retriever

    # 1) Try to execute compiled_graph if it's runnable
    if compiled_graph is not None:
        for meth in ("run", "execute", "__call__", "call"):
            func = getattr(compiled_graph, meth, None)
            if callable(func):
                try:
                    result = func(state)
                    if isinstance(result, dict):
                        final_state = result
                    else:
                        final_state = getattr(result, "state", None) or getattr(result, "result", None) or result
                    if isinstance(final_state, dict):
                        answer = final_state.get("final_answer")
                        _append_history(final_state, user_query, answer)
                        return answer, final_state
                except Exception:
                    break

    # 2) Fallback procedural execution using graph node functions
    state = router_node(state)

    if state.get("retriever") is None and retriever is not None:
        state["retriever"] = retriever

    if state.get("retriever") is None:
        raise ValueError("No retriever available for procedural execution. Provide a retriever or a runnable compiled_graph.")

    state = retrieval_node(state)

    next_node = determine_next_node(state)
    if next_node == "summarization_node":
        if llm is None:
            def _simple_llm(prompt: str) -> str:
                return (prompt or "")[:200]

            llm = _simple_llm
        state = summarization_node(state, llm)
    else:
        if llm is None:
            def _simple_llm(prompt: str) -> str:
                return "I don't know"

            llm = _simple_llm
        state = generation_node(state, llm)

    answer = state.get("final_answer")
    _append_history(state, user_query, answer)
    return answer, state


def main() -> None:
    """
    Top-level runtime bootstrap for the Research RAG system.

    Responsibilities (comment-only):
    - Load .env variables using `python-dotenv`.
    - Configure LangSmith tracing environment variables (ensure LANGSMITH_TRACING is 'true').
    - Execute `run_ingestion_pipeline` to create and persist the Chroma index and obtain a retriever.
    - Load the main generation LLM via `load_hf_pipeline` (local Hugging Face pipeline).
      * Note: When targeting TensorFlow, ensure the chosen model has TF-compatible
        weights (e.g., TFAuto* checkpoints). Configure TensorFlow GPU support
        (CUDA/cuDNN) as appropriate and consider `tf.keras.mixed_precision`
        to improve throughput on supported hardware.
    - Call `build_research_rag_graph(llm, retriever)` to obtain a compiled graph.
    - Start `run_rag_chat_loop(compiled_graph)` to accept queries interactively.

    Safety & Operational Notes (comment-only):
    - Users must ensure the chosen LLM and embedding models are available locally
      or can be downloaded with the provided HUGGINGFACEHUB_API_TOKEN.
    - When running locally on GPU with TensorFlow, verify TensorFlow, CUDA and
      cuDNN versions are compatible with your GPU and the installed TensorFlow wheel.
    """
    # Load env
    load_dotenv()
    _activate_langsmith()

    # Run ingestion to get a retriever
    retriever = run_ingestion_pipeline()

    # Load the LLM — provider and model resolved from env (LLM_PROVIDER, LLM_MODEL_ID)
    llm = None
    try:
        llm = load_llm()
    except Exception:
        llm = None

    # Build the graph if possible
    compiled = build_research_rag_graph(llm=llm, retriever=retriever)

    # Run interactive loop (not for tests)
    try:
        run_rag_chat_loop(compiled, retriever=retriever, llm=llm)
    except NotImplementedError:
        # Expected in test environments where interactive loop is not desired
        return


if __name__ == "__main__":
    main()
