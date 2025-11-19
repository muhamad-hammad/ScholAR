"""Lightweight LLM adapter helpers.

Provides `adapt_llm` which returns a simple callable `fn(prompt) -> str`
that normalizes common HF pipeline objects, user callables, and simple
wrapper objects so the rest of the codebase can call LLMs uniformly.
"""
from typing import Callable, Any


def adapt_llm(llm: Any) -> Callable[[str], str] | None:
    """Return a callable that accepts a prompt string and returns text.

    - If `llm` is already callable, return it.
    - If `llm` is a Hugging Face `pipeline` object, wrap it and extract
      `generated_text` or string outputs.
    - If `llm` exposes `generate`, try to call it and stringify the result.
    - If `llm` is None, return None.
    """
    if llm is None:
        return None

    # If it's already a callable that accepts a prompt, use it directly.
    if callable(llm):
        return llm

    # Try to detect transformers' Pipeline without importing heavy modules
    try:
        from transformers.pipelines.base import Pipeline
    except Exception:
        Pipeline = None

    if Pipeline is not None and isinstance(llm, Pipeline):
        def _hf_pipeline(prompt: str) -> str:
            out = llm(prompt)
            # typical HF text-generation returns a list of dicts with 'generated_text'
            try:
                if isinstance(out, list) and len(out) and isinstance(out[0], dict):
                    return out[0].get("generated_text") or str(out[0])
                if isinstance(out, dict) and "generated_text" in out:
                    return out.get("generated_text")
                if isinstance(out, str):
                    return out
                # fallback to str
                return str(out)
            except Exception:
                return str(out)

        return _hf_pipeline

    # Generic wrapper for objects exposing `generate` or similar
    if hasattr(llm, "generate"):
        def _generate(prompt: str) -> str:
            try:
                out = llm.generate(prompt)
                if isinstance(out, dict) and "text" in out:
                    return out["text"]
                return str(out)
            except Exception:
                return str(llm)

        return _generate

    # Best-effort: return a function that stringifies the object with the prompt
    def _fallback(prompt: str) -> str:
        return str(llm)

    return _fallback
