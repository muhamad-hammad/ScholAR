from typing import Any
from langgraph.graph import StateGraph, END
from src.core_config import ResearchRAGState
from src.graph_nodes import router_node, retrieval_node, summarization_node, generation_node, determine_next_node
from langchain_core.runnables import Runnable


def build_research_rag_graph(llm: Any, retriever: Any) -> Any:
    """
    Build and compile the LangGraph StateGraph for the agentic RAG system.

    High-level design and implementation notes (comment-only):
    - Initialize StateGraph with the typed schema `ResearchRAGState` which ensures
      every node receives and returns a consistent state shape.
    - Add functional nodes to the graph, providing lightweight wrappers around
      the pure functions defined in `src.graph_nodes`.
    - Set `router_node` as the entry point for all executions.
    - Use `add_conditional_edges` from the router node to direct flows:
        * 'QNA' -> 'retrieval_node'
        * 'SUMMARY' -> 'summarization_node'
    - Define sequential edges for the QNA path:
        retrieval_node -> generation_node -> END
    - Define the summarization path to go directly to END after summarization_node.
    - Compile the graph and return the compiled/ready-to-run graph object.

    Observability notes:
    - Ensure that LangSmith tracing is enabled in environment variables before executing
      the compiled graph so that each node's input/output state is recorded.
    - Instrument prompts and model calls to include short, descriptive names for
      easier trace analysis in LangSmith.
    """

    # Try to build a proper LangGraph StateGraph if the library is available.
    try:
        graph = StateGraph(ResearchRAGState)

        # Wrap nodes so the call signature matches what StateGraph expects
        graph.add_node("router_node", router_node)

        # retrieval_node expects only state
        graph.add_node("retrieval_node", retrieval_node)

        # summarization and generation expect (state, llm) -> state, so wrap them
        graph.add_node("summarization_node", lambda state: summarization_node(state, llm))
        graph.add_node("generation_node", lambda state: generation_node(state, llm))

        graph.set_entry_point("router_node")

        # Add conditional edges from router using our determine_next_node function
        # determine_next_node will return the node name (e.g. 'retrieval_node' or 'summarization_node')
        graph.add_conditional_edges("router_node", determine_next_node, {"QNA": "retrieval_node", "SUMMARY": "summarization_node"})

        # QNA path
        graph.add_edge("retrieval_node", "generation_node")
        graph.add_edge("generation_node", END)

        # Summary path
        graph.add_edge("summarization_node", END)

        return graph.compile()
    except Exception as e:
        # If LangGraph isn't available or compile fails, surface a clear error
        raise RuntimeError("Failed to build StateGraph. Ensure 'langgraph' is installed and compatible. Original error: {}".format(e)) from e
