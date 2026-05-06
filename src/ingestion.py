# Ingestion module: document loaders, token-aware splitters, and vectorstore wiring.

from typing import List, Any
from langchain_community.document_loaders import DedocFileLoader
from langchain_text_splitters import TokenTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def _set_source(docs: List[Document], pdf_path: str) -> List[Document]:
    for d in docs:
        if not getattr(d, "metadata", None):
            d.metadata = {"source": pdf_path}
        else:
            d.metadata.setdefault("source", pdf_path)
    return docs


def load_documents(pdf_path: str, prefer_dedoc: bool = True) -> List[Document]:
    """
    Load and parse PDF research documents with structure preservation.

    Loader priority:
    1. DedocFileLoader — best table and layout preservation
    2. PyMuPDFLoader   — fast, accurate, no heavy system deps
    3. PyPDFLoader     — pure-Python last resort
    """
    if prefer_dedoc:
        try:
            loader = DedocFileLoader(pdf_path, with_tables=True)
            docs = loader.load()
            return _set_source(docs, pdf_path)
        except FileNotFoundError:
            raise
        except Exception:
            pass

    # PyMuPDF (fitz) — installed, no pdfminer dependency
    try:
        from langchain_community.document_loaders import PyMuPDFLoader
        docs = PyMuPDFLoader(pdf_path).load()
        return _set_source(docs, pdf_path)
    except Exception:
        pass

    # Pure-Python pypdf fallback
    from langchain_community.document_loaders import PyPDFLoader
    docs = PyPDFLoader(pdf_path).load()
    return _set_source(docs, pdf_path)


def get_text_splitter(tokenizer_name: str, chunk_size: int = 1024, chunk_overlap: int = 128) -> TokenTextSplitter:
    """
    Instantiate a TokenTextSplitter tied to the Hugging Face tokenizer.

    Implementation guidance (comment-only):
    - Use TokenTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size, chunk_overlap)
      where `tokenizer` is loaded via `AutoTokenizer.from_pretrained(tokenizer_name)`.
    - Using the HF tokenizer ensures chunk boundaries respect actual token counts for
      the chosen LLM and prevents context-window overflow during generation.
    - If the HF tokenizer is unavailable, fallback to RecursiveCharacterTextSplitter
      with conservative chunk sizes.
    """
    # Lazy import tokenizer to avoid network/IO at module import time
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        # Use the HF-aware TokenTextSplitter when available
        try:
            return TokenTextSplitter.from_huggingface_tokenizer(
                tokenizer=tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        except Exception:
            # If the specialized constructor isn't available, fall back
            return TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except Exception:
        # If HF tokenizer isn't available locally, fallback to a character splitter
        return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def create_vectorstore(docs: List[Document], embeddings: Embeddings, persist_directory: str = None) -> Chroma:
    """
    Index split documents into a ChromaDB vector store.

    Implementation guidance (comment-only):
    - Convert Document objects into text chunks and embed them using the provided
      embeddings model in batched mode for efficiency.
    - Create a persistent Chroma instance with `persist_directory` set to CHROMA_PERSIST_DIR
      from environment variables to avoid re-ingestion between runs.
    - Store necessary metadata (document id, source, page numbers) alongside vectors
      to enable provenance reporting in generation_node.
    - Return the instantiated Chroma vectorstore object.
    """
    # Use Chroma.from_documents for straightforward indexing.
    # We avoid persistent side effects unless `persist_directory` is provided.
    try:
        if persist_directory:
            vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
            # If Chroma supports a persist() call, call it. Guarded to avoid import-time errors.
            try:
                vectordb.persist()
            except Exception:
                # Some wrappers persist automatically; ignore persistent-call failures.
                pass
        else:
            vectordb = Chroma.from_documents(documents=docs, embedding=embeddings)
        return vectordb
    except Exception as e:
        raise RuntimeError(f"Failed to create Chroma vectorstore: {e}") from e


def get_retriever(vectorstore: Chroma, k: int = 4) -> Any:
    """
    Convert a Chroma vectorstore into a LangChain Retriever configured for top-k.

    Implementation guidance (comment-only):
    - Expose parameters like `k`, `score_threshold`, and optionally use hybrid
      search configurations if available.
    - Return a Retriever or a function that accepts a query and returns top-k
      LangChain Document objects used by the retrieval_node.
    """
    # Prefer vectorstore.as_retriever(search_kwargs={}) which is the common pattern
    # in LangChain adapters. Fall back to other available methods and finally a
    # tiny wrapper around similarity_search if necessary.
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        return retriever
    except Exception:
        try:
            retriever = vectorstore.get_retriever(search_kwargs={"k": k})
            return retriever
        except Exception:
            # Last-resort simple retriever wrapper
            class SimpleRetriever:
                def __init__(self, vs, k):
                    self.vs = vs
                    self.k = k

                def get_relevant_documents(self, query: str):
                    # Many vectorstores expose similarity_search(query, k=k)
                    try:
                        return self.vs.similarity_search(query, k=self.k)
                    except TypeError:
                        # Some implementations accept (query, k) without keyword
                        return self.vs.similarity_search(query, self.k)

            return SimpleRetriever(vectorstore, k)
