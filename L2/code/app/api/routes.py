from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.chunking import split_text_into_chunks
from app.core.embeddings import embedding_service
from app.core.settings import settings
from app.db.database import get_db
from app.db.repository import insert_document_chunks, query_top_k
from app.schemas.documents import (
    AddDocumentRequest,
    AddDocumentResponse,
    QueryRequest,
    QueryResponse,
    QueryMatch,
)

router = APIRouter()


@router.post("/add_document", response_model=AddDocumentResponse)
def add_document(payload: AddDocumentRequest, db: Session = Depends(get_db)) -> AddDocumentResponse:
    if payload.chunk_overlap >= payload.chunk_size:
        raise HTTPException(status_code=400, detail="chunk_overlap must be smaller than chunk_size")

    chunks = split_text_into_chunks(
        payload.content,
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
    )
    if not chunks:
        raise HTTPException(status_code=400, detail="Document content is empty after normalization")

    vectors = embedding_service.encode_texts(chunks)
    inserted = insert_document_chunks(
        db,
        document_id=payload.document_id,
        source=payload.source,
        chunks=chunks,
        vectors=vectors,
    )
    return AddDocumentResponse(document_id=payload.document_id, chunk_count=inserted)


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, db: Session = Depends(get_db)) -> QueryResponse:
    if len(payload.query_vector) != settings.embedding_dimension:
        raise HTTPException(
            status_code=400,
            detail=f"query_vector length must be {settings.embedding_dimension}",
        )

    matches = query_top_k(db, query_vector=payload.query_vector, top_k=payload.top_k)
    results = [
        QueryMatch(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            source=chunk.source,
            content=chunk.content,
            distance=distance,
        )
        for chunk, distance in matches
    ]
    return QueryResponse(results=results)
