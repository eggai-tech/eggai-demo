from typing import Any

from pydantic import BaseModel, Field, field_validator


class PolicyDocument(BaseModel):
    id: str
    title: str
    text: str
    category: str
    chunk_index: int
    source_file: str
    relevance: float | None = None

    # Enhanced metadata fields
    page_numbers: list[int] = []
    page_range: str | None = None
    headings: list[str] = []
    citation: str | None = None

    # Relationships
    document_id: str | None = None
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None
    chunk_position: float | None = None


class SearchResponse(BaseModel):
    query: str
    category: str | None
    total_hits: int
    documents: list[PolicyDocument]
    search_type: str | None = None


class FullDocumentResponse(BaseModel):
    document_id: str
    category: str
    source_file: str
    full_text: str
    total_chunks: int
    total_characters: int
    total_tokens: int
    headings: list[str]
    page_numbers: list[int]
    page_range: str | None
    chunk_ids: list[str]
    metadata: dict[str, Any]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    category: str | None = None
    max_hits: int = Field(10, ge=1, le=100)
    search_type: str = Field("hybrid", pattern="^(vector|hybrid|keyword)$")  # "vector", "hybrid", or "keyword"

    @field_validator('query')
    @classmethod
    def validate_query_length(cls, v):
        if len(v) > 500:
            raise ValueError("Query too long. Maximum 500 characters allowed.")
        if not v.strip():
            raise ValueError("Query cannot be empty.")
        return v


class CategoryStats(BaseModel):
    name: str
    document_count: int


class ReindexRequest(BaseModel):
    force_rebuild: bool = False
    policy_ids: list[str] | None = None  # If None, reindex all

    def validate_policy_ids(self):
        """Validate policy IDs are valid categories."""
        if self.policy_ids:
            valid_ids = {"auto", "home", "health", "life"}
            invalid_ids = [pid for pid in self.policy_ids if pid not in valid_ids]
            if invalid_ids:
                raise ValueError(f"Invalid policy IDs: {invalid_ids}. Valid IDs are: {valid_ids}")
        return self


class ReindexResponse(BaseModel):
    status: str
    workflow_id: str
    total_documents_submitted: int
    policy_ids: list[str]


class PersonalPolicy(BaseModel):
    policy_number: str
    name: str
    policy_category: str
    premium_amount: float
    due_date: str
    status: str = "active"
    coverage_amount: float | None = None
    deductible: float | None = None


class PolicyListResponse(BaseModel):
    policies: list[PersonalPolicy]
    total: int
