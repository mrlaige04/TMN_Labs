from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.settings import settings


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=settings.default_top_k, ge=1, le=settings.max_top_k)


class SourceChunk(BaseModel):
    chunk_id: int
    document_id: str
    chunk_index: int
    source: str | None
    distance: float
    content: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    used_top_k: int
    sources: list[SourceChunk]
