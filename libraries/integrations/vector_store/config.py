from enum import StrEnum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VectorStoreBackend(StrEnum):
    FAISS = "faiss"
    VESPA = "vespa"


class VectorStoreConfig(BaseSettings):
    backend: VectorStoreBackend = Field(default=VectorStoreBackend.VESPA)
    embedding_dimensions: int = Field(default=384)
    max_hits: int = Field(default=10)
    ranking_profile: str = Field(default="default")
    faiss_path: Path = Field(default=Path(".data/vector-store/faiss"))

    model_config = SettingsConfigDict(
        env_prefix="VECTOR_STORE_", env_file=".env", env_ignore_empty=True, extra="ignore"
    )


vector_store_config = VectorStoreConfig()
