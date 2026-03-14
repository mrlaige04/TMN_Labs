"""Database access helpers."""

from __future__ import annotations

from typing import List

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import DocumentChunk


def insert_document_chunks(
    db: Session,
    *,
    document_id: str,
    source: str | None,
    chunks: List[str],
    vectors: List[List[float]],
) -> int:
    entities = [
        DocumentChunk(
            document_id=document_id,
            chunk_index=index,
            content=chunk_text,
            source=source,
            embedding=vector,
        )
        for index, (chunk_text, vector) in enumerate(zip(chunks, vectors))
    ]
    db.add_all(entities)
    db.commit()
    return len(entities)


def query_top_k(db: Session, *, query_vector: List[float], top_k: int) -> list[tuple[DocumentChunk, float]]:
    stmt = (
        select(DocumentChunk, DocumentChunk.embedding.cosine_distance(query_vector).label("distance"))
        .order_by("distance")
        .limit(top_k)
    )
    rows = db.execute(stmt).all()
    return [(row[0], float(row[1])) for row in rows]
