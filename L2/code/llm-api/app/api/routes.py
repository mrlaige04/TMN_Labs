from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.embeddings import embedding_service
from app.core.ollama_client import OllamaClientError, generate_answer
from app.core.prompting import build_prompt
from app.core.settings import settings
from app.db.database import get_db
from app.db.repository import retrieve_top_k_chunks
from app.schemas.qa import QueryRequest, QueryResponse, SourceChunk

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query_qa(payload: QueryRequest, db: Session = Depends(get_db)) -> QueryResponse:
    query_vector = embedding_service.encode_query(payload.question)
    if len(query_vector) != settings.embedding_dimension:
        raise HTTPException(
            status_code=500,
            detail=f"Embedding dimension mismatch, expected {settings.embedding_dimension}",
        )

    retrieved = retrieve_top_k_chunks(db, query_vector=query_vector, top_k=payload.top_k)
    if not retrieved:
        raise HTTPException(
            status_code=404,
            detail="No indexed chunks found. Add documents first using Task 1 API.",
        )

    context_chunks = [chunk.content for chunk, _ in retrieved]
    prompt = build_prompt(context_chunks=context_chunks, user_query=payload.question)
    try:
        answer = generate_answer(prompt)
    except OllamaClientError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    sources = [
        SourceChunk(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            source=chunk.source,
            distance=distance,
            content=chunk.content,
        )
        for chunk, distance in retrieved
    ]
    return QueryResponse(
        question=payload.question,
        answer=answer,
        used_top_k=payload.top_k,
        sources=sources,
    )
