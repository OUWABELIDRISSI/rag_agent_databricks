"""
RAGAS evaluation pipeline.
Evaluates RAG quality and logs results to PostgreSQL + Langfuse (optional).
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass

import httpx
import psycopg

from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvalResult:
    query: str
    answer: str
    contexts: list[str]
    faithfulness: float
    answer_relevancy: float
    context_recall: float
    latency_ms: int
    model: str


def _call_llm(system: str, user: str) -> str:
    """Call LLM via OpenRouter for evaluation scoring."""
    with httpx.Client(timeout=60) as client:
        response = client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.llm_model,
                "temperature": 0.0,
                "max_tokens": 256,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


def _score_faithfulness(answer: str, contexts: list[str]) -> float:
    """
    Faithfulness: are all claims in the answer supported by the context?
    Returns score between 0.0 and 1.0.
    """
    context_str = "\n\n".join(contexts)
    system = """You are an evaluation assistant. 
Given a context and an answer, score how faithful the answer is to the context.
A faithful answer only contains information present in the context.
Respond ONLY with a JSON: {"score": 0.0-1.0, "reason": "..."}"""

    user = f"""Context:
{context_str}

Answer:
{answer}"""

    try:
        raw = _call_llm(system=system, user=user)
        return float(json.loads(raw)["score"])
    except Exception:
        return 0.0


def _score_answer_relevancy(question: str, answer: str) -> float:
    """
    Answer relevancy: does the answer actually address the question?
    Returns score between 0.0 and 1.0.
    """
    system = """You are an evaluation assistant.
Given a question and an answer, score how relevant the answer is to the question.
A relevant answer directly addresses what was asked.
Respond ONLY with a JSON: {"score": 0.0-1.0, "reason": "..."}"""

    user = f"""Question: {question}

Answer: {answer}"""

    try:
        raw = _call_llm(system=system, user=user)
        return float(json.loads(raw)["score"])
    except Exception:
        return 0.0


def _score_context_recall(question: str, contexts: list[str]) -> float:
    """
    Context recall: do the retrieved contexts contain enough info to answer?
    Returns score between 0.0 and 1.0.
    """
    context_str = "\n\n".join(contexts)
    system = """You are an evaluation assistant.
Given a question and retrieved contexts, score whether the contexts contain
enough information to answer the question completely.
Respond ONLY with a JSON: {"score": 0.0-1.0, "reason": "..."}"""

    user = f"""Question: {question}

Contexts:
{context_str}"""

    try:
        raw = _call_llm(system=system, user=user)
        return float(json.loads(raw)["score"])
    except Exception:
        return 0.0


def evaluate(
    query: str,
    answer: str,
    contexts: list[str],
    latency_ms: int = 0,
) -> EvalResult:
    """Run full RAGAS-style evaluation on a single RAG response."""
    logger.info("evaluation_start", query_preview=query[:60])

    faithfulness = _score_faithfulness(answer, contexts)
    answer_relevancy = _score_answer_relevancy(query, answer)
    context_recall = _score_context_recall(query, contexts)

    result = EvalResult(
        query=query,
        answer=answer,
        contexts=contexts,
        faithfulness=faithfulness,
        answer_relevancy=answer_relevancy,
        context_recall=context_recall,
        latency_ms=latency_ms,
        model=settings.llm_model,
    )

    logger.info(
        "evaluation_done",
        faithfulness=round(faithfulness, 3),
        answer_relevancy=round(answer_relevancy, 3),
        context_recall=round(context_recall, 3),
    )

    _store_eval(result)
    return result


def _store_eval(result: EvalResult) -> None:
    """Persist evaluation result to PostgreSQL."""
    conn_str = settings.postgres_url.replace("+psycopg", "")
    try:
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO eval_traces (
                        id, query, answer, contexts,
                        faithfulness, answer_relevancy, context_recall,
                        latency_ms, model
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        str(uuid.uuid4()),
                        result.query,
                        result.answer,
                        result.contexts,
                        result.faithfulness,
                        result.answer_relevancy,
                        result.context_recall,
                        result.latency_ms,
                        result.model,
                    ),
                )
                conn.commit()
        logger.info("eval_stored")
    except Exception as e:
        logger.error("eval_store_failed", error=str(e))