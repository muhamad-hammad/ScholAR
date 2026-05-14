from typing import List
from main import run_rag_once


class CompiledGraphStub:
    def __init__(self, response: str):
        self.response = response

    def run(self, state: dict) -> dict:
        # Simulate a compiled graph that returns a final state dict
        return {"final_answer": self.response, "meta": {"from": "compiled"}}


class FakeDoc:
    def __init__(self, text: str, source: str = "doc1"):
        self.page_content = text
        self.metadata = {"source": source}


class FakeRetriever:
    def __init__(self, docs: List[FakeDoc]):
        self._docs = docs

    def get_relevant_documents(self, query: str):
        return self._docs


def test_run_rag_once_with_compiled_graph_stub():
    stub = CompiledGraphStub("COMPILED_ANSWER")
    ans, state = run_rag_once(stub, "Any query")
    assert ans == "COMPILED_ANSWER"
    assert state.get("meta", {}).get("from") == "compiled"


def test_run_rag_once_procedural_qna_with_fake_retriever_and_stub_llm():
    docs = [FakeDoc("This is content about methods and results.", source="paper.pdf")]
    retriever = FakeRetriever(docs)

    # Simple LLM stub that returns a deterministic answer when called
    def llm_stub(prompt: str) -> str:
        return "STUB_ANSWER"

    answer, state = run_rag_once(None, "What is the main contribution?", retriever=retriever, llm=llm_stub)
    assert answer == "STUB_ANSWER"
    assert isinstance(state, dict)
    # Ensure retrieved_docs were attached
    assert state.get("retrieved_docs") is not None


def test_run_rag_once_procedural_summary_with_fake_retriever_and_stub_llm():
    docs = [FakeDoc("Paragraph 1 about findings.", source="paper.pdf"), FakeDoc("Paragraph 2 about methods.", source="paper.pdf")]
    retriever = FakeRetriever(docs)

    # LLM stub that returns different tokens depending on whether it's asked to synthesize
    def llm_stub(prompt: str) -> str:
        p = (prompt or "").lower()
        if "abstract:" in p or "synthesize" in p:
            return "FINAL_ABSTRACT"
        return "INTERIM_SUMMARY"

    answer, state = run_rag_once(None, "Please summarize the paper", retriever=retriever, llm=llm_stub)
    assert answer == "FINAL_ABSTRACT"
    assert state.get("raw_summary_parts") is not None
    assert state.get("meta", {}).get("summarization", {}).get("parts") == len(state.get("raw_summary_parts"))
