"""API endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json

from src.agent.graph import run_agent
from src.api.schemas import AgentResponse, HealthResponse, QuestionRequest
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok", version="0.1.0")


@router.post("/ask", response_model=AgentResponse)
def ask(request: QuestionRequest) -> AgentResponse:
    """
    Ask a question to the RAG agent.
    Returns answer, sources, and routing decision.
    """
    logger.info("api_ask", question_preview=request.question[:60])
    start = time.perf_counter()

    try:
        result = run_agent(request.question)
    except Exception as e:
        logger.error("api_ask_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = int((time.perf_counter() - start) * 1000)
    logger.info("api_ask_done", latency_ms=latency_ms, route=result["route"])

    return AgentResponse(
        answer=result["answer"],
        sources=result["sources"],
        route=result["route"],
    )


@router.post("/ask/stream")
def ask_stream(request: QuestionRequest) -> StreamingResponse:
    """
    Streaming version of /ask.
    Returns Server-Sent Events (SSE) chunks as the answer is generated.
    """
    def generate():
        try:
            result = run_agent(request.question)
            # Stream word by word
            words = result["answer"].split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            # Send sources at the end
            yield f"data: {json.dumps({'sources': result['sources'], 'route': result['route'], 'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )