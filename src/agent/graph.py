"""
RAG Agent graph definition using LangGraph.

Flow:
  question → router → [rag: retriever → reranker → generator]
                     [direct: generator]
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import generator_node, reranker_node, retriever_node, router_node
from src.agent.state import AgentState
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _route_decision(state: AgentState) -> str:
    """Conditional edge: routes to 'rag' or 'direct' branch."""
    return state.get("route", "rag")


def build_graph() -> StateGraph:
    """Builds and compiles the RAG agent graph."""
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("router", router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("reranker", reranker_node)
    graph.add_node("generator", generator_node)

    # Entry point
    graph.add_edge(START, "router")

    # Conditional routing after router
    graph.add_conditional_edges(
        "router",
        _route_decision,
        {
            "rag": "retriever",
            "direct": "generator",
        },
    )

    # RAG pipeline edges
    graph.add_edge("retriever", "reranker")
    graph.add_edge("reranker", "generator")

    # Generator always leads to END
    graph.add_edge("generator", END)

    compiled = graph.compile()
    logger.info("graph_compiled")
    return compiled


# Singleton — import this in the API and elsewhere
agent_graph = build_graph()


def run_agent(question: str) -> dict:
    """
    Convenience function to run the agent on a single question.

    Returns:
        dict with keys: answer, sources, route
    """
    initial_state: AgentState = {
        "question": question,
        "route": "",
        "chunks": [],
        "relevant_chunks": [],
        "answer": "",
        "sources": [],
        "messages": [],
    }

    result = agent_graph.invoke(initial_state)

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "route": result["route"],
    }
