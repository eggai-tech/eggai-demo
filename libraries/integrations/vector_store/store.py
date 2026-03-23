import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from libraries.observability.logger import get_console_logger
from libraries.observability.tracing import create_tracer

from .base import VectorStoreBase
from .config import VectorStoreConfig
from .query_utils import parse_field_query
from .schemas import PolicyDocument

try:
    import faiss
except ImportError:
    faiss = None

logger = get_console_logger("vector_store")
tracer = create_tracer("vector_store", "client")


class InMemoryVectorStore(VectorStoreBase):
    """Local vector store with optional FAISS acceleration and on-disk state sharing."""

    def __init__(self, config: VectorStoreConfig | None = None):
        self.config = config or VectorStoreConfig()
        self._lock = threading.Lock()
        self._storage_path = Path(self.config.faiss_path)
        self._state_file = self._storage_path / "state.json"
        self._state_mtime = 0.0

        # Document storage: id -> document dict
        self._documents: dict[str, dict[str, Any]] = {}
        # Embeddings are stored separately so the vector index can be rebuilt across processes.
        self._embeddings: dict[str, list[float]] = {}
        self._index_ids: list[str] = []
        self._embedding_matrix: np.ndarray | None = None
        self._faiss_index: Any | None = None

        self._tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        self._tfidf_matrix = None
        self._tfidf_doc_ids: list[str] = []

        if faiss is None:
            logger.warning(
                "FAISS is not installed; using NumPy fallback for local vector search"
            )

        self._load_from_disk()

    def _clear_state(self) -> None:
        self._documents = {}
        self._embeddings = {}
        self._index_ids = []
        self._embedding_matrix = None
        self._faiss_index = None
        self._tfidf_matrix = None
        self._tfidf_doc_ids = []
        self._state_mtime = 0.0

    def _disk_state_mtime(self) -> float:
        if not self._state_file.exists():
            return 0.0
        return self._state_file.stat().st_mtime

    def _sync_from_disk_if_needed(self) -> None:
        disk_mtime = self._disk_state_mtime()
        if disk_mtime == 0.0:
            if self._state_mtime != 0.0:
                self._clear_state()
            return

        if disk_mtime <= self._state_mtime:
            return

        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if not self._state_file.exists():
            self._clear_state()
            return

        payload = json.loads(self._state_file.read_text(encoding="utf-8"))
        self._documents = payload.get("documents", {})
        self._embeddings = {
            doc_id: embedding
            for doc_id, embedding in payload.get("embeddings", {}).items()
            if embedding
        }
        self._rebuild_indices()
        self._state_mtime = self._disk_state_mtime()

    def _persist_to_disk(self) -> None:
        self._storage_path.mkdir(parents=True, exist_ok=True)

        payload = {
            "documents": self._documents,
            "embeddings": self._embeddings,
        }

        tmp_path = self._state_file.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        tmp_path.replace(self._state_file)
        self._state_mtime = self._disk_state_mtime()

    def _remove_persisted_state(self) -> None:
        if self._state_file.exists():
            self._state_file.unlink()
        self._state_mtime = 0.0

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
        try:
            self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(corpus)
        except ValueError:
            # This can happen if all content reduces to stop words.
            self._tfidf_matrix = None
            self._tfidf_doc_ids = []

    def _normalize_rows(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return matrix

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0, 1.0, norms)
        return matrix / safe_norms

    def _rebuild_vector_index(self) -> None:
        self._index_ids = []
        self._embedding_matrix = None
        self._faiss_index = None

        valid_embeddings: list[tuple[str, list[float]]] = []
        for doc_id, embedding in self._embeddings.items():
            if doc_id not in self._documents or not embedding:
                continue
            valid_embeddings.append((doc_id, embedding))

        if not valid_embeddings:
            return

        expected_dim = len(valid_embeddings[0][1])
        filtered_embeddings = [
            (doc_id, embedding)
            for doc_id, embedding in valid_embeddings
            if len(embedding) == expected_dim
        ]

        skipped_count = len(valid_embeddings) - len(filtered_embeddings)
        if skipped_count:
            logger.warning(
                f"Skipped {skipped_count} documents with inconsistent embedding dimensions"
            )

        if not filtered_embeddings:
            return

        self._index_ids = [doc_id for doc_id, _ in filtered_embeddings]
        matrix = np.array(
            [embedding for _, embedding in filtered_embeddings],
            dtype=np.float32,
        )
        self._embedding_matrix = self._normalize_rows(matrix)

        if faiss is None:
            return

        index = faiss.IndexFlatIP(expected_dim)
        index.add(self._embedding_matrix.copy())
        self._faiss_index = index

    def _rebuild_indices(self) -> None:
        """Rebuild vector and keyword indices from stored documents."""
        self._rebuild_vector_index()
        self._rebuild_tfidf()

    def _filter_by_category(self, doc_ids: list[str], category: str | None) -> list[str]:
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

    @tracer.start_as_current_span("check_connectivity")
    async def check_connectivity(self) -> bool:
        try:
            self._storage_path.mkdir(parents=True, exist_ok=True)
            logger.info("Local vector store connectivity check: OK")
            return True
        except Exception as e:
            logger.error(f"Local vector store connectivity check failed: {e}")
            return False

    @tracer.start_as_current_span("index_documents")
    async def index_documents(self, documents: list[PolicyDocument]) -> dict[str, Any]:
        logger.info(f"Starting indexing of {len(documents)} documents")

        success_count = 0
        error_count = 0
        errors = []

        with self._lock:
            self._sync_from_disk_if_needed()

            for doc in documents:
                try:
                    doc_dict = doc.to_dict()
                    self._documents[doc.id] = doc_dict

                    if doc.embedding:
                        self._embeddings[doc.id] = doc.embedding
                    else:
                        self._embeddings.pop(doc.id, None)

                    success_count += 1
                except Exception as e:
                    error_count += 1
                    error_msg = f"Failed to index document {doc.id}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            self._rebuild_indices()
            self._persist_to_disk()

        result = {
            "total_documents": len(documents),
            "successful": success_count,
            "failed": error_count,
            "errors": errors,
        }

        logger.info(f"Indexing completed: {success_count} successful, {error_count} failed")
        return result

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
            self._sync_from_disk_if_needed()

            if not self._documents:
                return []

            if not query:
                all_ids = self._filter_by_category(list(self._documents.keys()), category)
                results = self._format_results(all_ids[:max_hits], [1.0] * min(len(all_ids), max_hits))
                logger.info(f"Found {len(results)} documents")
                return results

            field_query = parse_field_query(query)
            if field_query:
                field, value = field_query
                results = self._search_by_field(field, value, category, max_hits)
                logger.info(f"Field search '{field}={value}' found {len(results)} documents")
                return results

            if self._tfidf_matrix is None:
                return []

            query_vec = self._tfidf_vectorizer.transform([query])
            scores = (self._tfidf_matrix @ query_vec.T).toarray().flatten()

            scored_ids = list(zip(self._tfidf_doc_ids, scores, strict=False))
            scored_ids.sort(key=lambda item: item[1], reverse=True)

            if category:
                scored_ids = [
                    (doc_id, score)
                    for doc_id, score in scored_ids
                    if self._documents.get(doc_id, {}).get("category") == category
                ]

            scored_ids = [(doc_id, score) for doc_id, score in scored_ids if score > 0]
            scored_ids = scored_ids[:max_hits]

            results = self._format_results(
                [doc_id for doc_id, _ in scored_ids],
                [score for _, score in scored_ids],
            )
            logger.info(f"Found {len(results)} documents")
            return results

    @tracer.start_as_current_span("get_document_count")
    async def get_document_count(self) -> int:
        with self._lock:
            self._sync_from_disk_if_needed()
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
        del ranking_profile
        max_hits = max_hits or self.config.max_hits

        logger.info(f"Performing vector search, category: '{category}', max_hits: {max_hits}")

        with self._lock:
            self._sync_from_disk_if_needed()

            if not self._index_ids or self._embedding_matrix is None:
                return []

            query_vec = np.array([query_embedding], dtype=np.float32)
            query_vec = self._normalize_rows(query_vec)

            if not np.any(query_vec):
                return []

            search_k = min(len(self._index_ids), max_hits * 3)

            if self._faiss_index is not None:
                distances, indices = self._faiss_index.search(query_vec, search_k)
                ranked = [
                    (float(score), int(idx))
                    for score, idx in zip(distances[0], indices[0], strict=False)
                ]
            else:
                scores = (self._embedding_matrix @ query_vec.T).flatten()
                ranked_indices = np.argsort(scores)[::-1][:search_k]
                ranked = [(float(scores[idx]), int(idx)) for idx in ranked_indices]

            results = []
            for score, idx in ranked:
                if idx < 0 or idx >= len(self._index_ids):
                    continue

                doc_id = self._index_ids[idx]
                doc = self._documents.get(doc_id)
                if not doc:
                    continue
                if category and doc.get("category") != category:
                    continue

                results.append({**doc, "relevance": score})
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

        keyword_results = await self.search_documents(
            query=query, category=category, max_hits=max_hits * 2
        )
        vector_results = await self.vector_search(
            query_embedding=query_embedding, category=category, max_hits=max_hits * 2
        )

        keyword_scores: dict[str, float] = {}
        vector_scores: dict[str, float] = {}
        all_docs: dict[str, dict[str, Any]] = {}

        for result in keyword_results:
            doc_id = result["id"]
            keyword_scores[doc_id] = result.get("relevance", 0.0)
            all_docs[doc_id] = result

        for result in vector_results:
            doc_id = result["id"]
            vector_scores[doc_id] = result.get("relevance", 0.0)
            all_docs[doc_id] = result

        def normalize(scores: dict[str, float]) -> dict[str, float]:
            if not scores:
                return scores
            max_score = max(scores.values())
            min_score = min(scores.values())
            score_range = max_score - min_score
            if score_range == 0:
                return dict.fromkeys(scores, 1.0)
            return {
                key: (value - min_score) / score_range
                for key, value in scores.items()
            }

        norm_keyword = normalize(keyword_scores)
        norm_vector = normalize(vector_scores)

        hybrid_scored = []
        for doc_id in set(norm_keyword) | set(norm_vector):
            keyword_score = norm_keyword.get(doc_id, 0.0)
            vector_score = norm_vector.get(doc_id, 0.0)
            hybrid_score = alpha * vector_score + (1 - alpha) * keyword_score
            hybrid_scored.append((doc_id, hybrid_score))

        hybrid_scored.sort(key=lambda item: item[1], reverse=True)
        hybrid_scored = hybrid_scored[:max_hits]

        results = []
        for doc_id, score in hybrid_scored:
            doc = all_docs.get(doc_id, {})
            results.append({**doc, "relevance": score})

        logger.info(f"Hybrid search found {len(results)} documents")
        return results

    async def get_document(self, doc_id: str, **kwargs) -> dict[str, Any] | None:
        del kwargs
        with self._lock:
            self._sync_from_disk_if_needed()
            return self._documents.get(doc_id)

    async def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            self._sync_from_disk_if_needed()

            if doc_id not in self._documents:
                return False

            del self._documents[doc_id]
            self._embeddings.pop(doc_id, None)
            self._rebuild_indices()
            self._persist_to_disk()
            return True

    async def delete_all_documents(self) -> int:
        with self._lock:
            self._sync_from_disk_if_needed()
            count = len(self._documents)
            self._clear_state()
            self._remove_persisted_state()
            return count
