from typing import TypedDict, List, Any
import os
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document

from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from transformers import TFAutoModelForCausalLM, AutoTokenizer, pipeline


class ResearchRAGState(TypedDict):
    """Shared mutable state passed between LangGraph nodes."""
    user_query: str
    query_intent: str
    retrieved_docs: List[Any]
    raw_summary_parts: List[str]
    final_answer: str
    chat_history: List[Any]


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
