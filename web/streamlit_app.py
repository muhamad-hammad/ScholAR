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
import io
import os
import tempfile
import traceback

import streamlit as st
from dotenv import load_dotenv

from main import run_ingestion_pipeline, run_rag_once, _activate_langsmith
from src.workflow_builder import build_research_rag_graph
from src.core_config import load_llm

load_dotenv()
_activate_langsmith()


def save_uploaded_pdf(uploaded_file) -> str:
    suffix = ".pdf"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return tmp_path


st.set_page_config(page_title="ResearchPaperReader RAG", layout="wide")

# ── session state defaults ────────────────────────────────────────────────────
for _k, _v in {
    "retriever": None,
    "compiled_graph": None,
    "llm": None,
    "ingested_file": None,
    "conversation_history": [],
    "usage_mode": None,   # "browse" | "default_key" | "own_key"
    "active_view": "chat",   # "chat" | "summary" | "visuals"
    "paper_summary": None,
    "pdf_bytes": None,
    "pdf_images": None,
    "theme_mode": "dark",   # "dark" | "light"
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── theme palettes ────────────────────────────────────────────────────────────
_THEMES = {
    "dark": {
        "bg": "#0F172A",
        "sidebar_bg": "#1E293B",
        "secondary_bg": "#1E293B",
        "text": "#E2E8F0",
        "muted_text": "#94A3B8",
        "accent": "#7C3AED",
        "accent_hover": "#8B5CF6",
        "accent_text": "#FFFFFF",
        "border": "#334155",
        "btn_bg": "#1E293B",
        "btn_hover_bg": "#334155",
    },
    "light": {
        "bg": "#FAFAFA",
        "sidebar_bg": "#F1F5F9",
        "secondary_bg": "#F1F5F9",
        "text": "#0F172A",
        "muted_text": "#475569",
        "accent": "#6D28D9",
        "accent_hover": "#7C3AED",
        "accent_text": "#FFFFFF",
        "border": "#E2E8F0",
        "btn_bg": "#FFFFFF",
        "btn_hover_bg": "#EDE9FE",
    },
}


def _inject_theme_css(mode: str) -> None:
    p = _THEMES.get(mode, _THEMES["dark"])
    css = f"""
    <style>
      :root {{
        --app-bg: {p['bg']};
        --app-sidebar-bg: {p['sidebar_bg']};
        --app-secondary-bg: {p['secondary_bg']};
        --app-text: {p['text']};
        --app-muted: {p['muted_text']};
        --app-accent: {p['accent']};
        --app-accent-hover: {p['accent_hover']};
        --app-accent-text: {p['accent_text']};
        --app-border: {p['border']};
        --app-btn-bg: {p['btn_bg']};
        --app-btn-hover-bg: {p['btn_hover_bg']};
      }}
      [data-testid="stAppViewContainer"],
      [data-testid="stHeader"],
      .stApp {{
        background-color: var(--app-bg) !important;
        color: var(--app-text) !important;
      }}
      [data-testid="stSidebar"],
      [data-testid="stSidebarContent"] {{
        background-color: var(--app-sidebar-bg) !important;
      }}
      [data-testid="stSidebar"] * {{
        color: var(--app-text);
      }}
      [data-testid="stAppViewContainer"] h1,
      [data-testid="stAppViewContainer"] h2,
      [data-testid="stAppViewContainer"] h3,
      [data-testid="stAppViewContainer"] h4,
      [data-testid="stAppViewContainer"] p,
      [data-testid="stAppViewContainer"] label,
      [data-testid="stAppViewContainer"] span,
      [data-testid="stAppViewContainer"] li {{
        color: var(--app-text);
      }}
      [data-testid="stCaptionContainer"], .st-emotion-cache-10trblm small {{
        color: var(--app-muted) !important;
      }}
      .stButton > button {{
        background-color: var(--app-btn-bg);
        color: var(--app-text);
        border: 1px solid var(--app-border);
        transition: background-color 120ms ease, border-color 120ms ease;
      }}
      .stButton > button:hover {{
        background-color: var(--app-btn-hover-bg);
        border-color: var(--app-accent);
        color: var(--app-text);
      }}
      .stButton > button[kind="primary"],
      .stButton > button[data-testid="baseButton-primary"] {{
        background-color: var(--app-accent) !important;
        color: var(--app-accent-text) !important;
        border-color: var(--app-accent) !important;
      }}
      .stButton > button[kind="primary"]:hover,
      .stButton > button[data-testid="baseButton-primary"]:hover {{
        background-color: var(--app-accent-hover) !important;
        border-color: var(--app-accent-hover) !important;
      }}
      .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {{
        background-color: var(--app-secondary-bg) !important;
        color: var(--app-text) !important;
        border-color: var(--app-border) !important;
      }}
      [data-testid="stChatMessage"] {{
        background-color: var(--app-secondary-bg);
        border: 1px solid var(--app-border);
      }}
      a {{ color: var(--app-accent); }}
      hr {{ border-color: var(--app-border) !important; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


_inject_theme_css(st.session_state.theme_mode)

# ── provider / model config ───────────────────────────────────────────────────
_PROVIDERS = ["huggingface", "openai", "google", "groq", "openrouter", "grok"]
_DEFAULT_MODELS = {
    "huggingface": os.getenv("LLM_MODEL_ID", ""),
    "openai": os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),
    "google": os.getenv("LLM_MODEL_ID", "gemini-1.5-flash"),
    "groq": os.getenv("LLM_MODEL_ID", "llama-3.1-8b-instant"),
    "openrouter": os.getenv("LLM_MODEL_ID", "openai/gpt-4o-mini"),
    "grok": os.getenv("LLM_MODEL_ID", "grok-3-mini"),
}
_API_KEY_VARS = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "grok": "XAI_API_KEY",
}


# ── welcome dialog ────────────────────────────────────────────────────────────
@st.dialog("Welcome to ResearchPaperReader RAG", width="large")
def welcome_dialog():
    st.markdown(
        "Before we get started — how would you like to use the AI assistant?"
    )
    st.write("")

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown("#### Browse only")
        st.caption(
            "Explore and retrieve passages from your documents. "
            "No AI answer generation — no API key needed."
        )
        if st.button("Continue without a key", use_container_width=True):
            st.session_state.usage_mode = "browse"
            st.rerun()

    with col2:
        st.markdown("#### Use configured key")
        st.caption(
            "An API key is already set in your `.env` file. "
            "We'll load it automatically — nothing extra needed."
        )
        if st.button("Use my .env key", use_container_width=True):
            st.session_state.usage_mode = "default_key"
            st.rerun()

    with col3:
        st.markdown("#### Enter my own key")
        st.caption(
            "Paste a fresh API key directly in the sidebar. "
            "Useful when testing a different account or model."
        )
        if st.button("I'll provide a key", use_container_width=True):
            st.session_state.usage_mode = "own_key"
            st.rerun()


# Show dialog on first load
if st.session_state.usage_mode is None:
    welcome_dialog()
    st.stop()

# ── page header ───────────────────────────────────────────────────────────────
st.title("ResearchPaperReader RAG")

_MODE_LABELS = {
    "browse": "Browse only (no AI generation)",
    "default_key": "Using configured .env key",
    "own_key": "Using a custom API key",
}
st.caption(f"Mode: {_MODE_LABELS.get(st.session_state.usage_mode, '')}")

# ── sidebar ───────────────────────────────────────────────────────────────────
ingest_btn = False
with st.sidebar:
    st.header("Setup")

    _is_dark = st.toggle(
        "Dark mode",
        value=(st.session_state.theme_mode == "dark"),
        key="theme_toggle",
    )
    _new_mode = "dark" if _is_dark else "light"
    if _new_mode != st.session_state.theme_mode:
        st.session_state.theme_mode = _new_mode
        st.rerun()

    st.divider()

    # Let user switch mode without reloading the page
    if st.button("Change startup mode", use_container_width=True):
        st.session_state.usage_mode = None
        st.session_state.llm = None
        st.session_state.compiled_graph = None
        st.rerun()

    st.divider()
    st.subheader("Document")
    uploaded = st.file_uploader("Upload a research PDF to index", type=["pdf"])

    st.subheader("AI model")
    env_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    default_idx = _PROVIDERS.index(env_provider) if env_provider in _PROVIDERS else 1
    provider = st.selectbox("Provider", _PROVIDERS, index=default_idx)

    llm_model = st.text_input("Model ID", value=_DEFAULT_MODELS.get(provider, ""))

    api_key = ""
    if st.session_state.usage_mode != "browse" and provider in _API_KEY_VARS:
        env_key_name = _API_KEY_VARS[provider]
        env_val = os.getenv(env_key_name, "")

        if st.session_state.usage_mode == "default_key":
            # Show the key name but keep it hidden; value comes from .env
            st.text_input(
                f"{env_key_name} (from .env)",
                value=env_val,
                type="password",
                disabled=True,
            )
            api_key = env_val
        else:
            # own_key — editable
            api_key = st.text_input(
                env_key_name,
                value=env_val,
                type="password",
            )

    if st.session_state.usage_mode == "browse":
        st.info("Browse mode: AI answer generation is disabled.")

    ingest_btn = st.button("Run ingestion", use_container_width=True)

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.conversation_history = []
        st.session_state.compiled_graph = None
        st.rerun()

# ── ingestion & LLM loading ───────────────────────────────────────────────────
if ingest_btn and uploaded is not None:
    if st.session_state.ingested_file == uploaded.name:
        st.info("This file is already indexed — using the cached retriever.")
    else:
        st.info("Saving PDF and running ingestion — this may take a moment.")
        st.session_state.pdf_bytes = bytes(uploaded.getbuffer())
        st.session_state.paper_summary = None
        st.session_state.pdf_images = None
        tmp_pdf = save_uploaded_pdf(uploaded)
        try:
            os.environ["PDF_INPUT_PATH"] = tmp_pdf
            with st.spinner("Indexing document..."):
                retriever = run_ingestion_pipeline()
            st.success("Ingestion complete — retriever is ready.")
            st.session_state.retriever = retriever
            st.session_state.ingested_file = uploaded.name
        except Exception as e:
            st.error("Ingestion failed: %s" % str(e))
            st.code(traceback.format_exc())
        finally:
            try:
                os.remove(tmp_pdf)
            except OSError:
                pass

    if st.session_state.usage_mode != "browse":
        try:
            if api_key and provider in _API_KEY_VARS:
                os.environ[_API_KEY_VARS[provider]] = api_key
            os.environ["LLM_PROVIDER"] = provider
            if llm_model:
                os.environ["LLM_MODEL_ID"] = llm_model
            with st.spinner(f"Loading {provider} model ({llm_model or 'default'})..."):
                llm = load_llm(provider=provider, model_id=llm_model or None)
            st.success("Model loaded (%s)" % provider)
            st.session_state.llm = llm
            st.session_state.compiled_graph = None
        except ImportError as e:
            msg = str(e)
            pkg = None
            for candidate in ("langchain-groq", "langchain-openai", "langchain-google-genai", "langchain_huggingface", "transformers"):
                if candidate in msg or candidate.replace("-", "_") in msg:
                    pkg = candidate.replace("_", "-")
                    break
            hint = f"\n\nInstall it with: `pip install {pkg}`" if pkg else ""
            st.error(f"Missing dependency for the **{provider}** provider.\n\n{msg}{hint}")
            st.session_state.llm = None
        except ValueError as e:
            st.error(f"Configuration error for the **{provider}** provider: {e}")
            st.session_state.llm = None
        except Exception as e:
            st.warning("Could not load the model — AI generation will be unavailable. Error: %s" % e)
            st.session_state.llm = None

# ── status banner ─────────────────────────────────────────────────────────────
if st.session_state.retriever is None:
    st.info("No document indexed yet. Upload a PDF and click **Run ingestion**.")
else:
    st.success("Document indexed and ready.")

# ── helpers ───────────────────────────────────────────────────────────────────
def _ensure_compiled_graph() -> bool:
    """Build the compiled workflow graph if missing. Returns True if usable."""
    if st.session_state.compiled_graph is not None:
        return True
    try:
        with st.spinner("Building workflow graph..."):
            st.session_state.compiled_graph = build_research_rag_graph(
                llm=st.session_state.llm,
                retriever=st.session_state.retriever,
            )
        return True
    except Exception as e:
        st.warning("Could not build the workflow graph; falling back to procedural mode. Error: %s" % e)
        st.session_state.compiled_graph = None
        return False


def _extract_pdf_images(pdf_bytes: bytes) -> list:
    """Return a list of dicts {image: bytes, page: int, ext: str} extracted from the PDF.

    Uses PyMuPDF (fitz) when available — it ships with the project's PDF stack.
    Returns an empty list and an error string in a tuple form via the caller.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF (`pymupdf`) is required for visualizations. "
            "Install it with `pip install pymupdf`."
        ) from exc

    images: list = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_index, page in enumerate(doc, start=1):
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                try:
                    base = doc.extract_image(xref)
                except Exception:
                    continue
                images.append(
                    {
                        "image": base.get("image"),
                        "ext": base.get("ext", "png"),
                        "page": page_index,
                    }
                )
    return images


# ── view switcher (Chat / Summary / Visualizations) ───────────────────────────
_VIEWS = [("chat", "Chat"), ("summary", "Summary"), ("visuals", "Visualizations")]
nav_cols = st.columns(len(_VIEWS))
for col, (view_key, label) in zip(nav_cols, _VIEWS):
    is_active = st.session_state.active_view == view_key
    if col.button(
        ("● " + label) if is_active else label,
        key="nav_%s" % view_key,
        use_container_width=True,
        type="primary" if is_active else "secondary",
    ):
        st.session_state.active_view = view_key
        st.rerun()

st.markdown("---")

# ── view: Chat ────────────────────────────────────────────────────────────────
if st.session_state.active_view == "chat":
    st.header("Conversation")
    for turn in st.session_state.conversation_history:
        with st.chat_message(turn.get("role", "user")):
            st.write(turn.get("content", ""))

    st.header("Ask a question")
    query = st.text_input("Enter your question", value="")
    ask = st.button("Ask")

    if ask:
        if not query:
            st.warning("Please enter a question.")
        elif st.session_state.retriever is None:
            st.warning("Please index a document first.")
        elif st.session_state.usage_mode == "browse" and st.session_state.llm is None:
            st.warning(
                "You are in Browse-only mode. Switch the startup mode to "
                "**Use configured key** or **Enter my own key** to enable AI answers."
            )
        else:
            _ensure_compiled_graph()
            with st.spinner("Generating answer..."):
                try:
                    answer, state = run_rag_once(
                        st.session_state.compiled_graph,
                        query,
                        retriever=st.session_state.retriever,
                        llm=st.session_state.llm,
                        conversation_history=st.session_state.conversation_history,
                    )
                    st.session_state.conversation_history = state.get(
                        "conversation_history", st.session_state.conversation_history
                    )
                    st.rerun()
                except Exception as e:
                    st.error("Error while generating answer: %s" % e)
                    st.code(traceback.format_exc())

# ── view: Summary ─────────────────────────────────────────────────────────────
elif st.session_state.active_view == "summary":
    st.header("Paper summary")
    st.caption(
        "An AI-generated abstract built from the indexed paper. "
        "The summary is cached — click **Regenerate** to recompute."
    )

    can_summarize = (
        st.session_state.retriever is not None
        and not (st.session_state.usage_mode == "browse" and st.session_state.llm is None)
    )

    btn_label = "Regenerate summary" if st.session_state.paper_summary else "Generate summary"
    if st.button(btn_label, disabled=not can_summarize):
        _ensure_compiled_graph()
        summary_query = (
            "Provide a comprehensive summary of this paper covering its motivation, "
            "methods, key findings, and conclusions."
        )
        with st.spinner("Summarizing the paper..."):
            try:
                answer, _state = run_rag_once(
                    st.session_state.compiled_graph,
                    summary_query,
                    retriever=st.session_state.retriever,
                    llm=st.session_state.llm,
                    conversation_history=[],
                )
                st.session_state.paper_summary = answer or "(no summary returned)"
            except Exception as e:
                st.error("Error while generating summary: %s" % e)
                st.code(traceback.format_exc())

    if st.session_state.retriever is None:
        st.info("Index a document first to generate a summary.")
    elif st.session_state.usage_mode == "browse" and st.session_state.llm is None:
        st.warning(
            "Browse-only mode is active. Switch startup mode to enable "
            "AI summary generation."
        )

    if st.session_state.paper_summary:
        st.markdown("### Summary")
        st.write(st.session_state.paper_summary)

# ── view: Visualizations ──────────────────────────────────────────────────────
elif st.session_state.active_view == "visuals":
    st.header("Paper visualizations")
    st.caption("Figures, charts, and other images extracted from the indexed PDF.")

    if st.session_state.pdf_bytes is None:
        st.info("Upload a PDF and run ingestion to see its figures here.")
    else:
        if st.session_state.pdf_images is None:
            with st.spinner("Extracting images from the PDF..."):
                try:
                    st.session_state.pdf_images = _extract_pdf_images(
                        st.session_state.pdf_bytes
                    )
                except RuntimeError as e:
                    st.error(str(e))
                    st.session_state.pdf_images = []
                except Exception as e:
                    st.error("Error while extracting images: %s" % e)
                    st.code(traceback.format_exc())
                    st.session_state.pdf_images = []

        images = st.session_state.pdf_images or []
        if not images:
            st.info("No embedded images were found in this PDF.")
        else:
            st.success(f"Found {len(images)} image(s) in the PDF.")
            cols_per_row = 2
            for row_start in range(0, len(images), cols_per_row):
                row = images[row_start : row_start + cols_per_row]
                cols = st.columns(len(row))
                for col, img in zip(cols, row):
                    with col:
                        st.image(
                            io.BytesIO(img["image"]),
                            caption=f"Page {img['page']}",
                            use_container_width=True,
                        )

            if st.button("Re-extract images"):
                st.session_state.pdf_images = None
                st.rerun()

st.markdown("---")
st.markdown("Run with: `streamlit run web/streamlit_app.py`")
