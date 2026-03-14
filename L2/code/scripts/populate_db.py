"""Populate vector index from local text files."""

from __future__ import annotations

from pathlib import Path

from app.core.chunking import split_text_into_chunks
from app.core.embeddings import embedding_service
from app.db.database import SessionLocal, init_database
from app.db.repository import insert_document_chunks


def main() -> None:
    init_database()
    data_dir = Path(__file__).resolve().parents[1] / "data"
    files = sorted(data_dir.glob("*.txt"))
    if not files:
        print("No .txt files found in data directory.")
        return

    with SessionLocal() as db:
        for file_path in files:
            content = file_path.read_text(encoding="utf-8")
            chunks = split_text_into_chunks(content)
            vectors = embedding_service.encode_texts(chunks)
            inserted = insert_document_chunks(
                db,
                document_id=file_path.stem,
                source=str(file_path.name),
                chunks=chunks,
                vectors=vectors,
            )
            print(f"Indexed {file_path.name}: {inserted} chunks")


if __name__ == "__main__":
    main()
