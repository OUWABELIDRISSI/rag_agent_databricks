"""
LangGraph nodes for the RAG agent.

Nodes:
  - router    : decides 'rag' vs 'direct'
  - retriever : fetches relevant chunks from pgvector
  - reranker  : filters low-relevance chunks
  - generator : calls Claude to produce the final answer
"""

from __future__ import annotations

import json
import httpx
#import anthropic
from langchain_core.messages import HumanMessage

from src.agent.prompts import DIRECT_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT, ROUTER_PROMPT
from src.agent.retriever import VectorRetriever
from src.agent.state import AgentState
from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Shared Anthropic client (thread-safe, reuse across calls)
#_client = anthropic.Anthropic(api_key=settings.openrouter_api_key)
_retriever = VectorRetriever(top_k=6)

def _call_llm(system: str, user: str) -> str:
    """Call any LLM via OpenRouter (OpenAI-compatible API)."""
    with httpx.Client(timeout=60) as client:
        response = client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/OUWABELIDRISSI",
            },
            json={
                "model": settings.llm_model,
                "temperature": settings.llm_temperature,
                "max_tokens": settings.llm_max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

# ── Node 1: Router ────────────────────────────────────────────────────────────

def router_node(state: AgentState) -> AgentState:
    """Decides whether to use RAG or answer directly."""
    question = state["question"]
    logger.info("router_node", question_preview=question[:60])

    raw = _call_llm(system=ROUTER_PROMPT, user=question)

    try:
        decision = json.loads(raw)
        route = decision.get("route", "rag")
    except json.JSONDecodeError:
        logger.warning("router_json_parse_error", raw=raw)
        route = "rag"  # safe default

    logger.info("router_decision", route=route)
    return {**state, "route": route}


# ── Node 2: Retriever ─────────────────────────────────────────────────────────

def retriever_node(state: AgentState) -> AgentState:
    """Fetches top-k relevant chunks from pgvector."""
    question = state["question"]
    logger.info("retriever_node", question_preview=question[:60])

    chunks = _retriever.retrieve(query=question)

    logger.info("retriever_node_done", chunks_found=len(chunks))
    return {**state, "chunks": chunks}



# ── Node 3: Reranker ──────────────────────────────────────────────────────────

def reranker_node(state: AgentState) -> AgentState:
    """Filters and reranks chunks by relevance score."""
    chunks = state["chunks"]

    ranked = sorted(chunks, key=lambda c: c.score, reverse=True)
    relevant = [c for c in ranked if c.score >= 0.40][:4]

    if not relevant and ranked:
        relevant = ranked[:2]

    logger.info(
        "reranker_node",
        input_chunks=len(chunks),
        kept_chunks=len(relevant),
        top_score=round(relevant[0].score, 3) if relevant else 0,
    )
    return {**state, "relevant_chunks": relevant}


# ── Node 4: Generator ─────────────────────────────────────────────────────────

def generator_node(state: AgentState) -> AgentState:
    """Calls LLM via OpenRouter to generate the final answer."""
    question = state["question"]
    route = state.get("route", "rag")

    if route == "rag":
        relevant_chunks = state.get("relevant_chunks", [])

        if not relevant_chunks:
            return {
                **state,
                "answer": (
                    "I couldn't find relevant information in the documentation "
                    "to answer your question. Please try rephrasing."
                ),
                "sources": [],
                "messages": [HumanMessage(content=question)],
            }

        context_parts = []
        for i, chunk in enumerate(relevant_chunks, 1):
            context_parts.append(
                f"[{i}] Source: {chunk.title} ({chunk.source})\n{chunk.content}"
            )
        context = "\n\n---\n\n".join(context_parts)

        system_prompt = RAG_SYSTEM_PROMPT.format(context=context)
        sources = list({c.source for c in relevant_chunks})

    else:
        system_prompt = DIRECT_SYSTEM_PROMPT
        sources = []

    logger.info("generator_node", route=route, question_preview=question[:60])

    answer = _call_llm(system=system_prompt, user=question)

    logger.info("generator_node_done", answer_length=len(answer))

    return {
        **state,
        "answer": answer,
        "sources": sources,
        "messages": [HumanMessage(content=question)],
    }