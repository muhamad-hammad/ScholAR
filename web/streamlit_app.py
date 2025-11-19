"""
Lightweight Streamlit UI for ResearchPaperReaderRAG

Features:
- Upload a PDF and run the ingestion pipeline (creates a retriever)
- Optionally provide an LLM model id (will attempt to load via `src.core_config`)
- Build the compiled graph and run single-shot queries via `run_rag_once`

Notes:
- This app prefers local models and will fall back to a simple stub LLM
  if model loading fails (keeps the UI responsive without heavy downloads).
"""
import os
import tempfile
import traceback

import streamlit as st

from main import run_ingestion_pipeline, run_rag_once
from src.workflow_builder import build_research_rag_graph
from src.core_config import load_hf_pipeline


def save_uploaded_pdf(uploaded_file) -> str:
    """Save uploaded Streamlit file to a temporary path and return the path."""
    suffix = ".pdf"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return tmp_path


st.set_page_config(page_title="ResearchPaperReader RAG", layout="wide")

st.title("ResearchPaperReader RAG — Streamlit UI")

if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "compiled_graph" not in st.session_state:
    st.session_state.compiled_graph = None
if "llm" not in st.session_state:
    st.session_state.llm = None

with st.sidebar:
    st.header("Ingestion & Model")
    uploaded = st.file_uploader("Upload a research PDF to index", type=["pdf"])
    llm_model = st.text_input("Optional LLM model id (local HF)", value="")
    ingest_btn = st.button("Run ingestion")

if ingest_btn and uploaded is not None:
    st.info("Saving uploaded PDF and running ingestion — this may take some time.")
    try:
        tmp_pdf = save_uploaded_pdf(uploaded)
        # Set env var so run_ingestion_pipeline picks it up
        os.environ["PDF_INPUT_PATH"] = tmp_pdf
        with st.spinner("Running ingestion pipeline..."):
            retriever = run_ingestion_pipeline()
        st.success("Ingestion completed — retriever ready")
        st.session_state.retriever = retriever
    except Exception as e:
        st.error("Ingestion failed: %s" % str(e))
        st.exception(traceback.format_exc())

    # Try to load LLM if requested
    if llm_model:
        try:
            with st.spinner(f"Loading LLM: {llm_model}..."):
                llm = load_hf_pipeline(llm_model)
            st.success("LLM loaded")
            st.session_state.llm = llm
        except Exception as e:
            st.warning("Failed to load LLM locally — falling back to stub LLM. Error: %s" % e)
            st.session_state.llm = None

if st.session_state.retriever is None:
    st.info("No retriever is available yet. Upload a PDF and click 'Run ingestion'.")
else:
    st.success("Retriever available")

st.header("Query")
query = st.text_input("Enter your question or ask for a summary", value="")
ask = st.button("Ask")

if ask:
    if not query:
        st.warning("Please enter a query.")
    elif st.session_state.retriever is None:
        st.warning("Please run ingestion first or provide a retriever.")
    else:
        # Build compiled graph if not present
        if st.session_state.compiled_graph is None:
            try:
                with st.spinner("Building workflow graph..."):
                    compiled = build_research_rag_graph(llm=st.session_state.llm, retriever=st.session_state.retriever)
                st.session_state.compiled_graph = compiled
            except Exception as e:
                st.warning("Could not build compiled graph; will run procedural fallback. Error: %s" % e)
                st.session_state.compiled_graph = None

        with st.spinner("Running RAG..."):
            try:
                answer, state = run_rag_once(st.session_state.compiled_graph, query, retriever=st.session_state.retriever, llm=st.session_state.llm)
                st.subheader("Assistant Answer")
                st.write(answer)

                st.subheader("Debug State")
                st.json(state)
            except Exception as e:
                st.error("Error while running RAG: %s" % e)
                st.exception(traceback.format_exc())

st.markdown("---")
st.markdown("Usage: run `streamlit run web/streamlit_app.py` and open the UI in your browser.")
