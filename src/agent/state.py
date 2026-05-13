"""
State definition for the RAG LangGraph agent.
Each field is updated by nodes as the graph executes.
"""

from __future__ import annotations

from typing import Annotated

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from src.agent.retriever import RetrievedChunk


class AgentState(TypedDict):
    """Shared state passed between all graph nodes."""

    # User input
    question: str

    # Router decision: 'rag' | 'direct'
    route: str

    # Retrieved chunks from pgvector
    chunks: list[RetrievedChunk]

    # Reranked / filtered chunks
    relevant_chunks: list[RetrievedChunk]

    # Final generated answer
    answer: str

    # Sources cited in the answer
    sources: list[str]

    # Conversation history (append-only via add_messages)
    messages: Annotated[list, add_messages]


class RouterDecision(BaseModel):
    """Structured output from the router node."""

    route: str = Field(
        description="'rag' if the question is about Databricks/Spark/dbt, "
                    "'direct' otherwise."
    )
    reason: str = Field(description="Short explanation of the routing decision.")