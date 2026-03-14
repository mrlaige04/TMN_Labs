from dataclasses import dataclass

import config


@dataclass(frozen=True)
class Settings:
    database_url: str = config.DATABASE_URL
    ollama_base_url: str = config.OLLAMA_BASE_URL
    ollama_model_name: str = config.OLLAMA_MODEL_NAME
    embedding_model_name: str = config.EMBEDDING_MODEL_NAME
    embedding_dimension: int = config.EMBEDDING_DIMENSION
    default_top_k: int = config.DEFAULT_TOP_K
    max_top_k: int = config.MAX_TOP_K


settings = Settings()
