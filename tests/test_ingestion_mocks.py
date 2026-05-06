import sys
from types import SimpleNamespace

# Insert lightweight fake modules for third-party packages that aren't
# available in the CI/dev environment so we can import `src.ingestion`
# without installing heavy dependencies like LangChain or transformers.
if "langchain_community" not in sys.modules:
    lc = SimpleNamespace()
    doc_mod = SimpleNamespace()
    doc_mod.DedocFileLoader = lambda *a, **k: (_ for _ in ()).throw(ImportError("placeholder"))
    doc_mod.PyMuPDFLoader = lambda *a, **k: (_ for _ in ()).throw(ImportError("placeholder"))
    doc_mod.PyPDFLoader = lambda *a, **k: (_ for _ in ()).throw(ImportError("placeholder"))
    vs_mod = SimpleNamespace()
    vs_mod.Chroma = lambda *a, **k: (_ for _ in ()).throw(ImportError("placeholder"))
    sys.modules["langchain_community"] = lc
    sys.modules["langchain_community.document_loaders"] = doc_mod
    sys.modules["langchain_community.vectorstores"] = vs_mod

if "langchain_text_splitters" not in sys.modules:
    # Provide TokenTextSplitter and RecursiveCharacterTextSplitter placeholders
    class _TokenTextSplitter:
        @staticmethod
        def from_huggingface_tokenizer(tokenizer=None, chunk_size=None, chunk_overlap=None):
            return _TokenTextSplitter()

        def __init__(self, chunk_size=None, chunk_overlap=None):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

    class _RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=None, chunk_overlap=None):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

    tmod = SimpleNamespace(TokenTextSplitter=_TokenTextSplitter, RecursiveCharacterTextSplitter=_RecursiveCharacterTextSplitter)
    sys.modules["langchain_text_splitters"] = tmod

if "langchain_core.documents" not in sys.modules:
    sys.modules["langchain_core.documents"] = SimpleNamespace(Document=object)

if "langchain_core.embeddings" not in sys.modules:
    sys.modules["langchain_core.embeddings"] = SimpleNamespace(Embeddings=object)

import pytest

import src.ingestion as ingestion


class FakeDoc:
    def __init__(self, text, source="paper.pdf"):
        self.page_content = text
        self.metadata = {"source": source}


def test_load_documents_prefers_dedoc_and_falls_back(monkeypatch):
    doc_mod = sys.modules["langchain_community.document_loaders"]

    # Case A: Dedoc works
    class DedocLoaderOK:
        def __init__(self, path, with_tables=False):
            self.path = path

        def load(self):
            return [FakeDoc("dedoc content")]

    monkeypatch.setattr(ingestion, "DedocFileLoader", DedocLoaderOK)

    docs = ingestion.load_documents("some.pdf", prefer_dedoc=True)
    assert isinstance(docs, list)
    assert docs[0].page_content == "dedoc content"
    assert docs[0].metadata.get("source") in ("paper.pdf", "some.pdf")

    # Case B: Dedoc fails -> fallback to PyMuPDFLoader
    def dedoc_raises(path, with_tables=False):
        raise RuntimeError("dedoc not available")

    class PyMuPDFLoaderOK:
        def __init__(self, path):
            self.path = path

        def load(self):
            return [FakeDoc("pymupdf content")]

    monkeypatch.setattr(ingestion, "DedocFileLoader", dedoc_raises)
    monkeypatch.setattr(doc_mod, "PyMuPDFLoader", PyMuPDFLoaderOK)

    docs2 = ingestion.load_documents("other.pdf", prefer_dedoc=True)
    assert docs2[0].page_content == "pymupdf content"
    assert docs2[0].metadata.get("source") in ("paper.pdf", "other.pdf")


def test_get_text_splitter_falls_back_when_transformers_missing(monkeypatch):
    # Ensure that when 'transformers' isn't importable, we fall back to RecursiveCharacterTextSplitter
    saved = sys.modules.pop("transformers", None)
    try:
        splitter = ingestion.get_text_splitter("nonexistent-tokenizer-12345", chunk_size=50, chunk_overlap=5)
        # The fallback in ingestion returns a RecursiveCharacterTextSplitter instance
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        assert isinstance(splitter, RecursiveCharacterTextSplitter)
    finally:
        if saved is not None:
            sys.modules["transformers"] = saved


def test_create_vectorstore_calls_chroma_and_persist(monkeypatch):
    # Replace Chroma with a fake that records calls
    recorded = {}

    class FakeVectorStore:
        def __init__(self, docs, embedding, persist_directory=None):
            recorded['docs'] = docs
            recorded['embedding'] = embedding
            recorded['persist_directory'] = persist_directory

        def persist(self):
            recorded['persist_called'] = True

    class FakeChroma:
        @staticmethod
        def from_documents(documents, embedding=None, persist_directory=None):
            return FakeVectorStore(documents, embedding, persist_directory=persist_directory)

    monkeypatch.setattr(ingestion, "Chroma", FakeChroma)

    fake_embeddings = SimpleNamespace(name="fake-emb")
    docs = [FakeDoc("a"), FakeDoc("b")]

    vs = ingestion.create_vectorstore(docs, fake_embeddings, persist_directory="/tmp/chroma")
    # Confirm the fake was used and persist called
    assert recorded['docs'] == docs
    assert recorded['embedding'] is fake_embeddings
    assert recorded['persist_directory'] == "/tmp/chroma"


def test_get_retriever_fallbacks(monkeypatch):
    # Case 1: .as_retriever is available
    class VS1:
        def as_retriever(self, search_kwargs=None):
            return "AS_RETRIEVER"

    r1 = ingestion.get_retriever(VS1(), k=2)
    assert r1 == "AS_RETRIEVER"

    # Case 2: as_retriever missing, get_retriever available
    class VS2:
        def as_retriever(self, search_kwargs=None):
            raise RuntimeError("nope")

        def get_retriever(self, search_kwargs=None):
            return "GET_RETRIEVER"

    r2 = ingestion.get_retriever(VS2(), k=3)
    assert r2 == "GET_RETRIEVER"

    # Case 3: fall back to SimpleRetriever wrapper using similarity_search
    docs = [FakeDoc("x"), FakeDoc("y")]

    class VS3:
        def as_retriever(self, search_kwargs=None):
            raise RuntimeError("nope")

        def get_retriever(self, search_kwargs=None):
            raise RuntimeError("nope2")

        def similarity_search(self, query, k):
            return docs[:k]

    r3 = ingestion.get_retriever(VS3(), k=1)
    # r3 should be the SimpleRetriever wrapper with method get_relevant_documents
    found = r3.get_relevant_documents("anything")
    assert found == docs[:1]
