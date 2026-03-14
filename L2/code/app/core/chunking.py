from __future__ import annotations
from typing import List


def split_text_into_chunks(
    text: str,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> List[str]:
    clean_text = " ".join(text.split())
    if not clean_text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: List[str] = []
    start = 0
    step = chunk_size - chunk_overlap
    text_len = len(clean_text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(clean_text[start:end])
        if end == text_len:
            break
        start += step

    return chunks
