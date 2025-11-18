from typing import List
from src.workflow_builder import build_research_rag_graph
from tests.test_run_rag_once import FakeDoc, FakeRetriever
from main import run_rag_once


class StubLLM:
    def __call__(self, prompt: str) -> str:
        return "COMPILED_STUB_ANSWER"


def test_compiled_graph_runs_with_stub_llm_and_fake_retriever():
    docs: List[FakeDoc] = [FakeDoc("Doc content about the study.", source="paper.pdf")]
    retriever = FakeRetriever(docs)

    llm = StubLLM()
    # build and compile the graph
    compiled = build_research_rag_graph(llm=llm, retriever=retriever)

    # run via run_rag_once which will call compiled.run(state)
    answer, state = run_rag_once(compiled, "What is the main result?", retriever=retriever)

    assert answer == "COMPILED_STUB_ANSWER"
    assert isinstance(state, dict)
    # Ensure generation metadata if present
    assert state.get("meta") is not None
