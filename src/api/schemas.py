"""Request and response schemas for the RAG API."""

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)


class AgentResponse(BaseModel):
    answer: str
    sources: list[str]
    route: str


class HealthResponse(BaseModel):
    status: str
    version: str