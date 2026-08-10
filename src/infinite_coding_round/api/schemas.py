"""Pydantic request/response models for the /ask API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language business question.")


class RetrievedSource(BaseModel):
    source: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    tools_used: list[str]
    generated_sql: str | None = None
    retrieved_sources: list[RetrievedSource]
    confidence: float
    validation_result: str
    validation_passed: bool
    retry_count: int
