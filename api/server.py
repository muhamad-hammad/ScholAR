import asyncio
import base64
import functools
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from main import _activate_langsmith, run_ingestion_pipeline, run_rag_once
from src.core_config import load_llm
from src.workflow_builder import build_research_rag_graph

try:
    _activate_langsmith()
except Exception:
    pass

# Prevent HuggingFace Hub from hanging indefinitely on slow/missing model downloads
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

_executor = ThreadPoolExecutor(max_workers=2)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_state: dict[str, Any] = {
    "retriever": None,
    "compiled_graph": None,
    "llm": None,
    "ingested_file": None,
}

_API_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "grok": "XAI_API_KEY",
}

_SUMMARY_QUESTION = (
    "Provide a comprehensive summary of this paper covering its motivation, "
    "methods, key findings, and conclusions."
)


_MIN_PX = 150  # skip icons/badges smaller than this in either dimension
# Unique-color threshold at a 128×128 thumbnail: logos/watermarks have flat areas
# and pass through with fewer distinct colours; rich figures (plots, photos, waveforms)
# easily exceed this number due to anti-aliasing, gradients, and data diversity.
_MIN_UNIQUE_COLORS = 200


def _looks_like_figure(img_bytes: bytes) -> bool:
    """Return True when the image has enough colour complexity to be a research figure.

    Uses a small thumbnail so the check is fast even for large embedded images.
    Falls back to True (keep the image) if PIL is unavailable.
    """
    try:
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.thumbnail((128, 128))
        # getcolors returns None when unique colours exceed maxcolors → complex image
        return img.getcolors(maxcolors=_MIN_UNIQUE_COLORS) is None
    except Exception:
        return True


def _extract_pdf_images(pdf_bytes: bytes) -> list:
    """Return a list of dicts {image: bytes, page: int, ext: str} extracted from the PDF.

    Uses PyMuPDF (fitz) when available — it ships with the project's PDF stack.
    Skips images smaller than _MIN_PX in either dimension, deduplicates by xref,
    and drops visually simple images (logos, watermarks) via colour-diversity check.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF (`pymupdf`) is required for visualizations. "
            "Install it with `pip install pymupdf`."
        ) from exc

    images: list = []
    seen_xrefs: set = set()
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_index, page in enumerate(doc, start=1):
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                if xref in seen_xrefs:
                    continue
                # img_info[2] = width, img_info[3] = height (pixels)
                w, h = img_info[2], img_info[3]
                if w < _MIN_PX or h < _MIN_PX:
                    continue
                seen_xrefs.add(xref)
                try:
                    base = doc.extract_image(xref)
                except Exception:
                    continue
                img_bytes = base.get("image")
                if not _looks_like_figure(img_bytes):
                    continue
                images.append(
                    {
                        "image": img_bytes,
                        "ext": base.get("ext", "png"),
                        "page": page_index,
                    }
                )
    return images


def _configure_llm(provider: str, model_id: str, api_key: str, usage_mode: str) -> Any:
    if usage_mode != "browse":
        env_var = _API_KEY_ENV.get(provider.lower())
        if env_var and api_key:
            os.environ[env_var] = api_key
        os.environ["LLM_PROVIDER"] = provider
        os.environ["LLM_MODEL_ID"] = model_id
        llm = load_llm(provider, model_id)
        _state["llm"] = llm
        _state["compiled_graph"] = None
        return llm
    return _state["llm"]


def _ensure_graph(llm: Any, retriever: Any) -> Any:
    if _state["compiled_graph"] is None:
        _state["compiled_graph"] = build_research_rag_graph(llm, retriever)
    return _state["compiled_graph"]


# ── models ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    conversation_history: list[dict[str, str]] = []
    provider: str = ""
    model_id: str = ""
    api_key: str = ""
    usage_mode: str = "browse"


class SummarizeRequest(BaseModel):
    provider: str = ""
    model_id: str = ""
    api_key: str = ""
    usage_mode: str = "browse"


# ── routes ────────────────────────────────────────────────────────────────────

_PROVIDER_DEFAULTS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "groq": "llama-3.3-70b-versatile",
    "openrouter": "meta-llama/llama-3.3-8b-instruct:free",
    "grok": "grok-3-mini",
}

_DEMO_PAPER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "research_paper.pdf")
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/env-config")
def env_config():
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model_id = os.getenv("LLM_MODEL_ID", "") or _PROVIDER_DEFAULTS.get(provider, "")
    env_var = _API_KEY_ENV.get(provider, "")
    has_key = bool(os.getenv(env_var)) if env_var else False
    return {"provider": provider, "model_id": model_id, "has_key": has_key}


_INGEST_TIMEOUT = 360  # seconds — generous for cold-start model downloads


async def _run_ingestion(pdf_path: str) -> object:
    """Run the blocking ingestion pipeline in a thread pool with a timeout."""
    os.environ["PDF_INPUT_PATH"] = pdf_path
    loop = asyncio.get_event_loop()
    try:
        retriever = await asyncio.wait_for(
            loop.run_in_executor(
                _executor, functools.partial(run_ingestion_pipeline, persist=False)
            ),
            timeout=_INGEST_TIMEOUT,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=(
                f"Ingestion timed out after {_INGEST_TIMEOUT}s. "
                "The backend may be downloading the embedding model for the first time — "
                "please wait 30 seconds and try again."
            ),
        ) from exc
    return retriever


@app.post("/ingest-demo")
async def ingest_demo():
    if not os.path.exists(_DEMO_PAPER_PATH):
        raise HTTPException(status_code=404, detail="Demo paper not found at data/research_paper.pdf.")
    try:
        retriever = await _run_ingestion(_DEMO_PAPER_PATH)
        _state["retriever"] = retriever
        _state["ingested_file"] = "research_paper.pdf"
        _state["compiled_graph"] = None
        return {"ok": True, "filename": "research_paper.pdf"}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ingest")
async def ingest(file: UploadFile):
    tmp_path = None
    try:
        suffix = os.path.splitext(file.filename or "upload.pdf")[1] or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        retriever = await _run_ingestion(tmp_path)
        _state["retriever"] = retriever
        _state["ingested_file"] = file.filename
        _state["compiled_graph"] = None
        return {"ok": True, "filename": file.filename}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/query")
async def query(req: QueryRequest):
    retriever = _state["retriever"]
    if retriever is None:
        raise HTTPException(status_code=400, detail="No document ingested yet.")

    try:
        llm = _configure_llm(req.provider, req.model_id, req.api_key, req.usage_mode)
        if llm is None:
            raise HTTPException(status_code=400, detail="LLM not configured.")
        graph = _ensure_graph(llm, retriever)
        answer, state = run_rag_once(graph, req.question, retriever, llm, req.conversation_history)
        return {"answer": answer, "conversation_history": state.get("conversation_history", [])}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/summarize")
async def summarize(req: SummarizeRequest):
    retriever = _state["retriever"]
    if retriever is None:
        raise HTTPException(status_code=400, detail="No document ingested yet.")

    try:
        llm = _configure_llm(req.provider, req.model_id, req.api_key, req.usage_mode)
        if llm is None:
            raise HTTPException(status_code=400, detail="LLM not configured.")
        graph = _ensure_graph(llm, retriever)
        answer, _ = run_rag_once(graph, _SUMMARY_QUESTION, retriever, llm, [])
        return {"summary": answer}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/images")
async def images(file: UploadFile):
    try:
        pdf_bytes = await file.read()
        extracted = _extract_pdf_images(pdf_bytes)
        result = [
            {
                "page": img["page"],
                "ext": img["ext"],
                "data": base64.b64encode(img["image"]).decode("utf-8"),
            }
            for img in extracted
            if img.get("image")
        ]
        return {"images": result}
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
