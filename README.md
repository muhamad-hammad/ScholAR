# ResearchPaperReaderRAG

[![CI](https://github.com/muhamad-hammad/ResearchPaperReaderRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/muhamad-hammad/ResearchPaperReaderRAG/actions/workflows/ci.yml)

Lightweight, agentic Retrieval-Augmented Generation (RAG) system for scientific research papers, built on LangChain + LangGraph with a Streamlit UI.

## Goals

- Zero-cost, open-source architecture using LangChain, LangGraph, LangSmith (free tier), and Hugging Face models (local or free endpoints).
- Two primary workflows: full-document summarization and targeted Q&A.

## Project Structure

```
src/            — Core library (ingestion, graph nodes, LLM config, workflow)
tests/          — Pytest suite with CI-safe mocks
web/            — Streamlit UI
scripts/        — CLI tools and launch scripts
data/           — Input PDF research papers
.github/        — CI/CD workflows
```

## Getting Started (PowerShell)

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Configure environment variables:
   - Copy `.env` and fill in your API keys (`HUGGINGFACEHUB_API_TOKEN`, `LANGSMITH_API_KEY`, model IDs).
   - Ensure `PDF_INPUT_PATH` points to a valid PDF (default: `./data/research_paper.pdf`).

4. Run the application:

```powershell
python main.py
```

## Streamlit UI

```powershell
.\scripts\run_app.ps1
```

Or directly:

```powershell
streamlit run web/streamlit_app.py
```

## CLI Usage

```powershell
python scripts/run_single_query.py --pdf data/research_paper.pdf --query "Summarize the paper"
```

## Running Tests

```powershell
pytest -q
```

For CI (minimal deps, no heavy ML libraries):

```powershell
pip install -r requirements-ci.txt
pytest -q
```

## TensorFlow Notes

This project supports TensorFlow for local model execution. If you have an NVIDIA GPU, install a TensorFlow wheel compatible with your CUDA/cuDNN versions. For CPU-only setups, use `tensorflow-cpu`.

`tensorflow-text` is optional and does not publish wheels for every Python/OS combination. If it fails to install, use a supported Python version (3.10/3.11) or rely on the pure-Python fallback.

## License

Provided for development. Ensure any models or data you download comply with their respective licenses.
