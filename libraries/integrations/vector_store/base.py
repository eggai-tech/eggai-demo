from abc import ABC, abstractmethod
from typing import Any

from .schemas import PolicyDocument


class VectorStoreBase(ABC):
    """Abstract base class for vector store implementations."""

    @abstractmethod
    async def check_connectivity(self) -> bool: ...

    @abstractmethod
    async def index_documents(self, documents: list[PolicyDocument]) -> dict[str, Any]: ...

    @abstractmethod
    async def search_documents(
        self,
        query: str,
        category: str | None = None,
        max_hits: int | None = None,
        ranking_profile: str | None = None,
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def get_document_count(self) -> int: ...

    @abstractmethod
    async def vector_search(
        self,
        query_embedding: list[float],
        category: str | None = None,
        max_hits: int | None = None,
        ranking_profile: str = "semantic",
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        category: str | None = None,
        max_hits: int | None = None,
        alpha: float = 0.7,
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def get_document(self, doc_id: str, **kwargs) -> dict[str, Any] | None: ...

    @abstractmethod
    async def delete_document(self, doc_id: str) -> bool: ...

    @abstractmethod
    async def delete_all_documents(self) -> int: ...
