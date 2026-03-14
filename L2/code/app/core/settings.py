from dataclasses import dataclass

import config


@dataclass(frozen=True)
class Settings:
    database_url: str = config.DATABASE_URL
    embedding_model_name: str = config.EMBEDDING_MODEL_NAME
    embedding_dimension: int = config.EMBEDDING_DIMENSION


settings = Settings()
