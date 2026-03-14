from __future__ import annotations

import requests

from app.core.settings import settings


class OllamaClientError(Exception):
    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def generate_answer(prompt: str) -> str:
    try:
        response = requests.post(
            f"{settings.ollama_base_url}/api/generate",
            json={
                "model": settings.ollama_model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=180,
        )
    except requests.RequestException as exc:
        raise OllamaClientError(f"Failed to connect to Ollama: {exc}", status_code=503) from exc

    if response.status_code >= 400:
        detail = response.text.strip()
        if not detail:
            detail = f"Ollama request failed with status {response.status_code}"
        raise OllamaClientError(detail, status_code=502)

    data = response.json()
    return str(data.get("response", "")).strip()
