from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import DocumentChunk


def retrieve_top_k_chunks(
    db: Session,
    *,
    query_vector: list[float],
    top_k: int,
) -> list[tuple[DocumentChunk, float]]:
    stmt = (
        select(DocumentChunk, DocumentChunk.embedding.cosine_distance(query_vector).label("distance"))
        .order_by("distance")
        .limit(top_k)
    )
    rows = db.execute(stmt).all()
    return [(row[0], float(row[1])) for row in rows]
