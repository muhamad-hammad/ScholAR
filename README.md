# ResearchPaperReaderRAG

[![CI](https://github.com/muhamad-hammad/ResearchPaperReaderRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/muhamad-hammad/ResearchPaperReaderRAG/actions/workflows/ci.yml)


Lightweight Research Paper RAG system with Streamlit UI and test-friendly design.

Quickstart
1. Install minimal dependencies for development:

```powershell
python -m pip install -r requirements-ci.txt
```

2. To run the Streamlit UI (optional install `streamlit`):

```powershell
python -m pip install streamlit
streamlit run web/streamlit_app.py
```

3. Run tests:

```powershell
python -m pytest -q
```

CLI usage

```powershell
python scripts/run_single_query.py --pdf example.pdf --query "Summarize the paper"
```
# ResearchPaperReaderRAG — Agentic RAG for Scientific Documents

Lightweight project skeleton for an open-source, agentic Retrieval-Augmented Generation (RAG)
system designed for scientific research papers. This repository contains structural stubs and
developer comments indicating how to implement ingestion, indexing, and an agentic LangGraph
workflow using LangChain and Hugging Face models (TensorFlow-first guidance in this branch).

Getting started (PowerShell)

1) Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Upgrade pip and install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3) Configure environment variables:

- Copy `.env` and replace placeholders (`HUGGINGFACEHUB_API_TOKEN`, `LANGSMITH_API_KEY`, model IDs).

4) Quick checks:

- Ensure `PDF_INPUT_PATH` in `.env` points to a valid PDF.
- Optionally create the `data/` directory and place a sample `research_paper.pdf`.

5) Run the project (after implementing the functions in `src/`):

```powershell
python main.py
```

Running tests

- A minimal pytest is included to check importability of key modules. Run with:

```powershell
pytest -q
```

Notes about TensorFlow and `tensorflow-text`

- This branch targets TensorFlow for local model execution. If you have an NVIDIA GPU,
  install a TensorFlow wheel compatible with your CUDA/cuDNN versions. For CPU-only setups,
  consider `tensorflow-cpu`.
- `tensorflow-text` is optional and does not publish wheels for every Python/OS combination
  (notably some Windows + newer Python releases). If `tensorflow-text` fails to install,
  either:
  - Use a supported Python version (e.g., 3.10/3.11) and install the matching wheel,
  - Build `tensorflow-text` from source (advanced), or
  - Use pure-Python text handling for preprocessing and avoid `tensorflow-text`.

Next steps

- Implement `src/ingestion.py` to create a robust, table-preserving PDF ingestion pipeline.
- Implement `src/core_config.py` loader functions to create Hugging Face TensorFlow pipelines
  or use TF-compatible checkpoints.
- Build and compile the LangGraph in `src/workflow_builder.py` and run sample traces in LangSmith.

License

- This skeleton is provided for development. Ensure that any models or data you download
  comply with their respective licenses.
# ResearchPaperReaderRAG — Agentic RAG for Scientific Documents

Lightweight project skeleton for an open-source, agentic Retrieval-Augmented Generation (RAG)
system designed for scientific research papers. This repository contains only structural
stubs and detailed developer comments indicating how to implement ingestion, indexing,
and an agentic LangGraph workflow using LangChain and Hugging Face local models.

Goals
- Provide a zero-cost, open-source architecture using LangChain, LangGraph, LangSmith (free tier),
  and Hugging Face models (local or free endpoints).
- Support two primary workflows: full-document summarization and targeted Q&A.

Getting started (PowerShell)
1) Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Upgrade pip and install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3) Configure environment variables:
- Copy `.env` and replace placeholders (HUGGINGFACEHUB_API_TOKEN, LANGSMITH_API_KEY, model IDs).

4) Quick checks:
- Ensure `PDF_INPUT_PATH` in `.env` points to a valid PDF.
- Optionally create the `data/` directory and place a sample `research_paper.pdf`.

5) Run the project (after implementing the functions in `src/`):

```powershell
python main.py
```

Running tests
- A minimal pytest is included to check importability of key modules. Run with:

```powershell
pytest -q
```

Notes and next steps
- This skeleton intentionally contains only imports, type signatures, and detailed comments
  describing responsibilities. Implement one module at a time (ingestion -> embeddings -> graph).
- For local LLMs, follow Hugging Face/transformers recommendations for quantization and
  `accelerate` configuration for your hardware (NVIDIA CUDA for GPU). See comments in `src/core_config.py`.
- Use LangSmith tracing (set `LANGSMITH_TRACING=true` in `.env`) to get visual traces for LangGraph runs.

License
- This skeleton is provided as-is for development. Ensure that any models or data you download
  comply with their respective licenses.
