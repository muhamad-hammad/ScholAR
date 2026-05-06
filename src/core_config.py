from __future__ import annotations

from typing import TypedDict, List, Any, Dict, Optional
import os
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document


class ResearchRAGState(TypedDict, total=False):
    """Shared mutable state passed between LangGraph nodes."""
    user_query: str
    query_intent: str
    retrieved_docs: List[Any]
    retrieval_metadata: List[Dict[str, Any]]
    raw_summary_parts: List[str]
    final_answer: str
    chat_history: List[Any]
    conversation_history: List[Dict[str, str]]
    retriever: Any
    meta: Dict[str, Any]


def load_hf_pipeline(model_id: str, task: str = "text-generation", **kwargs) -> HuggingFacePipeline:
    """Return a LangChain HuggingFacePipeline for `task` using `model_id`.

    Prefers TensorFlow backend when available; falls back to PyTorch. Extra
    generation kwargs (max_new_tokens, temperature, etc.) may be provided.
    """
    model = model_id or os.getenv("LLM_MODEL_ID")
    if not model:
        raise ValueError("No LLM model specified. Set LLM_MODEL_ID or pass model_id.")

    try:
        from transformers import pipeline as hf_pipeline, AutoTokenizer
    except Exception as e:
        raise ImportError("transformers is required to build a Hugging Face pipeline.") from e

    # Prefer TF if TFAutoModelForCausalLM is available
    use_tf = True
    try:
        from transformers import TFAutoModelForCausalLM  # type: ignore
    except Exception:
        use_tf = False

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    framework = "tf" if use_tf else "pt"
    hf_pipe = hf_pipeline(task, model=model, tokenizer=tokenizer, framework=framework, **kwargs)

    try:
        from langchain_huggingface.llms import HuggingFacePipeline as LC_HFPipeline
    except Exception as e:
        raise ImportError("langchain_huggingface is required to return a LangChain HuggingFacePipeline.") from e

    return LC_HFPipeline(pipeline=hf_pipe)


def load_llm(provider: str = None, model_id: str = None):
    """Return a callable ``fn(prompt: str) -> str`` for the requested provider.

    Provider is resolved from the ``provider`` arg, then ``LLM_PROVIDER`` env var,
    defaulting to ``huggingface``. Supported values: huggingface, openai, google.
    """
    provider = (provider or os.getenv("LLM_PROVIDER") or "huggingface").lower().strip()
    model_id = model_id or os.getenv("LLM_MODEL_ID") or ""

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set or is empty. "
                "Set it before using the openai provider."
            )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError("langchain-openai is required for the openai provider.") from e
        model = model_id or "gpt-4o-mini"
        chat = ChatOpenAI(model=model, api_key=api_key)

        def _openai_call(prompt: str) -> str:
            return chat.invoke(prompt).content

        return _openai_call

    if provider in ("google", "gemini"):
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is not set or is empty. "
                "Set it before using the google provider."
            )
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise ImportError("langchain-google-genai is required for the google provider.") from e
        model = model_id or "gemini-1.5-flash"
        chat = ChatGoogleGenerativeAI(model=model, google_api_key=api_key)

        def _google_call(prompt: str) -> str:
            return chat.invoke(prompt).content

        return _google_call

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set or is empty. "
                "Set it before using the groq provider."
            )
        try:
            from langchain_groq import ChatGroq
        except ImportError as e:
            raise ImportError("langchain-groq is required for the groq provider.") from e
        model = model_id or "llama-3.1-8b-instant"
        chat = ChatGroq(model=model, api_key=api_key)

        def _groq_call(prompt: str) -> str:
            return chat.invoke(prompt).content

        return _groq_call

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is not set or is empty. "
                "Set it before using the openrouter provider."
            )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError("langchain-openai is required for the openrouter provider.") from e
        model = model_id or "openai/gpt-4o-mini"
        chat = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        def _openrouter_call(prompt: str) -> str:
            return chat.invoke(prompt).content

        return _openrouter_call

    if provider in ("grok", "xai"):
        api_key = os.getenv("XAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "XAI_API_KEY environment variable is not set or is empty. "
                "Set it before using the grok provider."
            )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError("langchain-openai is required for the grok provider.") from e
        model = model_id or "grok-3-mini"
        chat = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )

        def _grok_call(prompt: str) -> str:
            return chat.invoke(prompt).content

        return _grok_call

    # Default: local Hugging Face pipeline
    return load_hf_pipeline(model_id)


def load_hf_embeddings(model_id: str):
    """Return a LangChain HuggingFaceEmbeddings instance for `model_id`.

    Falls back to the EMBEDDING_MODEL_ID env var when `model_id` is not provided.
    """
    model = model_id or os.getenv("EMBEDDING_MODEL_ID")
    if not model:
        raise ValueError("No embedding model specified. Set EMBEDDING_MODEL_ID or pass model_id.")

    try:
        from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    except Exception as e:
        raise ImportError("langchain_huggingface is required for Hugging Face embeddings.") from e

    return HuggingFaceEmbeddings(model_name=model)
