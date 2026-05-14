from typing import List
from src.graph_nodes import (
    router_node,
    determine_next_node,
    retrieval_node,
    summarization_node,
    generation_node,
)


class FakeDoc:
    def __init__(self, text: str, source: str = "doc1"):
        self.page_content = text
        self.metadata = {"source": source}


class FakeRetriever:
    def __init__(self, docs: List[FakeDoc]):
        self._docs = docs

    def get_relevant_documents(self, query: str):
        return self._docs


class StubLLM:
    """Very small deterministic LLM-like callable used for integration tests.

    The stub inspects the prompt to decide whether to return a short summary,
    a synthesized abstract, or a direct answer. It is intentionally simple and
    deterministic so it can run quickly in CI and on developer machines.
    """

    def __call__(self, prompt: str) -> str:
        p = (prompt or "").lower()
        if "summarize the following document chunk" in p or "summarize the following" in p:
            return "INTERIM_SUMMARY"
        if "synthesize them into a single concise abstract" in p or "abstract:" in p:
            return "FINAL_ABSTRACT"
        if "answer the user's question" in p or "user question" in p or "answer the user's question ONLY" in p:
            return "Final answer here. [Source: doc1]"
        # Default fallback
        return "ok"


def test_summary_and_qna_paths_work_with_stubs():
    # Prepare fake documents
    docs = [FakeDoc("This is a test paragraph about method and results.", source="paper.pdf")]

    # SUMMARY path
    state = {"user_query": "Please summarize the paper", "meta": {}}
    # router -> sets query_intent
    state = router_node(state)
    next_node = determine_next_node(state)
    assert next_node == "summarization_node"

    # set retrieved_docs to simulate retrieval step
    state["retrieved_docs"] = docs
    llm = StubLLM()
    state = summarization_node(state, llm)
    assert "final_answer" in state
    assert state.get("raw_summary_parts") is not None

    # QNA path
    retriever = FakeRetriever(docs)
    state2 = {"user_query": "What is the main contribution?", "retriever": retriever, "meta": {}}
    state2 = router_node(state2)
    next_node2 = determine_next_node(state2)
    assert next_node2 == "retrieval_node"

    # run retrieval_node (will use FakeRetriever)
    state2 = retrieval_node(state2)
    assert state2.get("retrieved_docs")
    # run generation_node with stub LLM
    state2 = generation_node(state2, llm)
    assert "final_answer" in state2
    assert "Source" in state2["final_answer"] or state2["meta"]["generation"]["sources_used"]
