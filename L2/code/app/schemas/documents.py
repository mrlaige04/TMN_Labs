from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class AddDocumentRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=128)
    content: str = Field(min_length=1)
    source: str | None = Field(default=None, max_length=512)
    chunk_size: int = Field(default=700, ge=100, le=3000)
    chunk_overlap: int = Field(default=120, ge=0, le=1000)


class AddDocumentResponse(BaseModel):
    document_id: str
    chunk_count: int


class QueryRequest(BaseModel):
    query_vector: List[float]
    top_k: int = Field(default=5, ge=1, le=50)


class QueryMatch(BaseModel):
    chunk_id: int
    document_id: str
    chunk_index: int
    source: str | None
    content: str
    distance: float


class QueryResponse(BaseModel):
    results: List[QueryMatch]
