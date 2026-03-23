from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PolicyDocument(BaseModel):
    id: str = Field(...)
    title: str = Field(...)
    text: str = Field(...)
    category: str = Field(...)
    chunk_index: int = Field(...)
    source_file: str = Field(...)
    page_numbers: list[int] = Field(default_factory=list)
    page_range: str | None = Field(None)
    headings: list[str] = Field(default_factory=list)
    char_count: int = Field(0)
    token_count: int = Field(0)
    document_id: str = Field(...)
    previous_chunk_id: str | None = Field(None)
    next_chunk_id: str | None = Field(None)
    chunk_position: float = Field(0.0)
    section_path: list[str] = Field(default_factory=list)
    embedding: list[float] | None = Field(None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "text": self.text,
            "category": self.category,
            "chunk_index": self.chunk_index,
            "source_file": self.source_file,
            "page_numbers": self.page_numbers,
            "page_range": self.page_range,
            "headings": self.headings,
            "char_count": self.char_count,
            "token_count": self.token_count,
            "document_id": self.document_id,
            "previous_chunk_id": self.previous_chunk_id,
            "next_chunk_id": self.next_chunk_id,
            "chunk_position": self.chunk_position,
            "section_path": self.section_path,
        }


class DocumentMetadata(BaseModel):
    id: str = Field(...)
    file_path: str = Field(...)
    file_name: str = Field(...)
    category: str = Field(...)
    total_pages: int = Field(0)
    total_chunks: int = Field(0)
    total_characters: int = Field(0)
    total_tokens: int = Field(0)
    document_type: str = Field("pdf")
    file_size: int = Field(0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_modified: datetime | None = Field(None)
    outline: list[dict[str, Any]] = Field(default_factory=list)
    key_sections: list[str] = Field(default_factory=list)
