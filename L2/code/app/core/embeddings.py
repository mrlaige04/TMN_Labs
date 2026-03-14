from __future__ import annotations

from threading import Lock
from typing import List

from sentence_transformers import SentenceTransformer

from app.core.settings import settings


class EmbeddingService:
    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None
        self._lock = Lock()

    def _get_model(self) -> SentenceTransformer:
        # Lazy-load model to speed up container startup.
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = SentenceTransformer(settings.embedding_model_name)
        return self._model

    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        model = self._get_model()
        vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return [vector.astype(float).tolist() for vector in vectors]


embedding_service = EmbeddingService()
