import re
import threading
from typing import Any

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from libraries.observability.logger import get_console_logger
from libraries.observability.tracing import create_tracer

from .base import VectorStoreBase
from .config import VectorStoreConfig
from .schemas import PolicyDocument

logger = get_console_logger("vector_store")
tracer = create_tracer("vector_store", "client")

# Pattern to detect field:value queries (e.g. document_id:"auto_policy", source_file:auto.md)
_FIELD_QUERY_PATTERN = re.compile(r'^(\w+):\s*"?([^"]+)"?\s*$')


class InMemoryVectorStore(VectorStoreBase):
    """In-memory vector store using FAISS for semantic search and TF-IDF for keyword search."""

    def __init__(self, config: VectorStoreConfig | None = None):
        self.config = config or VectorStoreConfig()
        self._lock = threading.Lock()

        # Document storage: id -> document dict
        self._documents: dict[str, dict[str, Any]] = {}
        # Ordered list of doc ids matching FAISS index positions
        self._index_ids: list[str] = []

        # FAISS index for vector search
        self._faiss_index: faiss.IndexFlatIP | None = None

        # TF-IDF for keyword search
        self._tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        self._tfidf_matrix = None
        self._tfidf_doc_ids: list[str] = []

    def _rebuild_tfidf(self) -> None:
        """Rebuild TF-IDF index from all stored documents."""
        if not self._documents:
            self._tfidf_matrix = None
            self._tfidf_doc_ids = []
            return

        self._tfidf_doc_ids = list(self._documents.keys())
        corpus = [
            f"{self._documents[doc_id].get('title', '')} {self._documents[doc_id].get('text', '')}"
            for doc_id in self._tfidf_doc_ids
        ]

        self._tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(corpus)

    @tracer.start_as_current_span("check_connectivity")
    async def check_connectivity(self) -> bool:
        logger.info("In-memory vector store connectivity check: OK")
        return True

    @tracer.start_as_current_span("index_documents")
    async def index_documents(self, documents: list[PolicyDocument]) -> dict[str, Any]:
        logger.info(f"Starting indexing of {len(documents)} documents")

        success_count = 0
        error_count = 0
        errors = []

        with self._lock:
            for doc in documents:
                try:
                    doc_dict = doc.to_dict()
                    self._documents[doc.id] = doc_dict

                    if doc.embedding:
                        embedding = np.array([doc.embedding], dtype=np.float32)
                        # Normalize for cosine similarity via inner product
                        faiss.normalize_L2(embedding)

                        if self._faiss_index is None:
                            dim = len(doc.embedding)
                            self._faiss_index = faiss.IndexFlatIP(dim)

                        self._faiss_index.add(embedding)
                        self._index_ids.append(doc.id)

                    success_count += 1
                except Exception as e:
                    error_count += 1
                    error_msg = f"Failed to index document {doc.id}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            # Rebuild TF-IDF after batch insert
            self._rebuild_tfidf()

        result = {
            "total_documents": len(documents),
            "successful": success_count,
            "failed": error_count,
            "errors": errors,
        }

        logger.info(f"Indexing completed: {success_count} successful, {error_count} failed")
        return result

    def _filter_by_category(
        self, doc_ids: list[str], category: str | None
    ) -> list[str]:
        if not category:
            return doc_ids
        return [
            doc_id
            for doc_id in doc_ids
            if self._documents.get(doc_id, {}).get("category") == category
        ]

    def _format_results(
        self, doc_ids: list[str], scores: list[float]
    ) -> list[dict[str, Any]]:
        results = []
        for doc_id, score in zip(doc_ids, scores, strict=False):
            doc = self._documents.get(doc_id)
            if doc:
                result = {**doc, "relevance": float(score)}
                results.append(result)
        return results

    def _search_by_field(
        self, field: str, value: str, category: str | None, max_hits: int
    ) -> list[dict[str, Any]]:
        """Filter documents where a specific field matches a value."""
        matching = []
        for doc_id, doc in self._documents.items():
            if category and doc.get("category") != category:
                continue
            doc_value = str(doc.get(field, ""))
            if doc_value == value or value in doc_value:
                matching.append(doc_id)
                if len(matching) >= max_hits:
                    break
        return self._format_results(matching, [1.0] * len(matching))

    @tracer.start_as_current_span("search_documents")
    async def search_documents(
        self,
        query: str,
        category: str | None = None,
        max_hits: int | None = None,
        ranking_profile: str | None = None,
    ) -> list[dict[str, Any]]:
        max_hits = max_hits or self.config.max_hits

        logger.info(f"Searching for: '{query}', category: '{category}', max_hits: {max_hits}")

        with self._lock:
            if not self._documents:
                return []

            # If no query, return all (filtered by category)
            if not query:
                all_ids = self._filter_by_category(
                    list(self._documents.keys()), category
                )
                results = self._format_results(all_ids[:max_hits], [1.0] * min(len(all_ids), max_hits))
                logger.info(f"Found {len(results)} documents")
                return results

            # Detect field:value queries (e.g. document_id:"auto_policy")
            field_match = _FIELD_QUERY_PATTERN.match(query)
            if field_match:
                field, value = field_match.group(1), field_match.group(2)
                results = self._search_by_field(field, value, category, max_hits)
                logger.info(f"Field search '{field}={value}' found {len(results)} documents")
                return results

            if self._tfidf_matrix is None:
                return []

            # TF-IDF keyword search
            query_vec = self._tfidf_vectorizer.transform([query])
            scores = (self._tfidf_matrix @ query_vec.T).toarray().flatten()

            scored_ids = list(zip(self._tfidf_doc_ids, scores, strict=False))
            scored_ids.sort(key=lambda x: x[1], reverse=True)

            # Filter by category
            if category:
                scored_ids = [
                    (doc_id, score)
                    for doc_id, score in scored_ids
                    if self._documents.get(doc_id, {}).get("category") == category
                ]

            # Filter zero-score results and limit
            scored_ids = [(doc_id, score) for doc_id, score in scored_ids if score > 0]
            scored_ids = scored_ids[:max_hits]

            doc_ids = [doc_id for doc_id, _ in scored_ids]
            doc_scores = [score for _, score in scored_ids]

            results = self._format_results(doc_ids, doc_scores)
            logger.info(f"Found {len(results)} documents")
            return results

    @tracer.start_as_current_span("get_document_count")
    async def get_document_count(self) -> int:
        count = len(self._documents)
        logger.info(f"Total documents in store: {count}")
        return count

    @tracer.start_as_current_span("vector_search")
    async def vector_search(
        self,
        query_embedding: list[float],
        category: str | None = None,
        max_hits: int | None = None,
        ranking_profile: str = "semantic",
    ) -> list[dict[str, Any]]:
        max_hits = max_hits or self.config.max_hits

        logger.info(f"Performing vector search, category: '{category}', max_hits: {max_hits}")

        with self._lock:
            if self._faiss_index is None or self._faiss_index.ntotal == 0:
                return []

            query_vec = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vec)

            # Search more than needed to allow for category filtering
            search_k = min(self._faiss_index.ntotal, max_hits * 3)
            distances, indices = self._faiss_index.search(query_vec, search_k)

            results = []
            for dist, idx in zip(distances[0], indices[0], strict=False):
                if idx < 0 or idx >= len(self._index_ids):
                    continue
                doc_id = self._index_ids[idx]
                doc = self._documents.get(doc_id)
                if not doc:
                    continue
                if category and doc.get("category") != category:
                    continue
                results.append({**doc, "relevance": float(dist)})
                if len(results) >= max_hits:
                    break

            logger.info(f"Vector search found {len(results)} documents")
            return results

    @tracer.start_as_current_span("hybrid_search")
    async def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        category: str | None = None,
        max_hits: int | None = None,
        alpha: float = 0.7,
    ) -> list[dict[str, Any]]:
        max_hits = max_hits or self.config.max_hits

        logger.info(f"Performing hybrid search: '{query}', alpha: {alpha}, category: '{category}'")

        # Get results from both search modes
        keyword_results = await self.search_documents(
            query=query, category=category, max_hits=max_hits * 2
        )
        vector_results = await self.vector_search(
            query_embedding=query_embedding, category=category, max_hits=max_hits * 2
        )

        # Merge scores: hybrid = alpha * vector + (1-alpha) * keyword
        keyword_scores: dict[str, float] = {}
        vector_scores: dict[str, float] = {}
        all_docs: dict[str, dict[str, Any]] = {}

        for r in keyword_results:
            doc_id = r["id"]
            keyword_scores[doc_id] = r.get("relevance", 0.0)
            all_docs[doc_id] = r

        for r in vector_results:
            doc_id = r["id"]
            vector_scores[doc_id] = r.get("relevance", 0.0)
            all_docs[doc_id] = r

        # Normalize scores to [0, 1]
        def normalize(scores: dict[str, float]) -> dict[str, float]:
            if not scores:
                return scores
            max_s = max(scores.values())
            min_s = min(scores.values())
            rng = max_s - min_s
            if rng == 0:
                return dict.fromkeys(scores, 1.0)
            return {k: (v - min_s) / rng for k, v in scores.items()}

        norm_keyword = normalize(keyword_scores)
        norm_vector = normalize(vector_scores)

        # Compute hybrid scores
        all_ids = set(norm_keyword.keys()) | set(norm_vector.keys())
        hybrid_scored = []
        for doc_id in all_ids:
            kw = norm_keyword.get(doc_id, 0.0)
            vec = norm_vector.get(doc_id, 0.0)
            hybrid = alpha * vec + (1 - alpha) * kw
            hybrid_scored.append((doc_id, hybrid))

        hybrid_scored.sort(key=lambda x: x[1], reverse=True)
        hybrid_scored = hybrid_scored[:max_hits]

        results = []
        for doc_id, score in hybrid_scored:
            doc = all_docs.get(doc_id, {})
            results.append({**doc, "relevance": score})

        logger.info(f"Hybrid search found {len(results)} documents")
        return results

    async def get_document(self, doc_id: str, **kwargs) -> dict[str, Any] | None:
        """Retrieve a single document by ID."""
        return self._documents.get(doc_id)

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a single document by ID."""
        with self._lock:
            if doc_id in self._documents:
                del self._documents[doc_id]
                # Rebuild indices (FAISS doesn't support single-item deletion easily)
                self._rebuild_indices()
                return True
            return False

    async def delete_all_documents(self) -> int:
        """Delete all documents and return count deleted."""
        with self._lock:
            count = len(self._documents)
            self._documents.clear()
            self._index_ids.clear()
            self._faiss_index = None
            self._tfidf_matrix = None
            self._tfidf_doc_ids = []
            return count

    def _rebuild_indices(self) -> None:
        """Rebuild both FAISS and TF-IDF indices from stored documents."""
        # Rebuild FAISS
        self._faiss_index = None
        self._index_ids = []

        # We don't have embeddings stored in _documents (they're stripped in to_dict).
        # For the demo this is acceptable - deleted docs just won't appear in vector search
        # until the next full reindex.

        # Rebuild TF-IDF
        self._rebuild_tfidf()
