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

    def to_vespa_dict(self) -> dict[str, Any]:
        data = {
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

        if self.embedding:
            data["embedding"] = {
                "cells": [
                    {"address": {"x": str(i)}, "value": float(v)}
                    for i, v in enumerate(self.embedding)
                ]
            }

        return data


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

    def to_vespa_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "file_path": self.file_path,
            "file_name": self.file_name,
            "category": self.category,
            "total_pages": self.total_pages,
            "total_chunks": self.total_chunks,
            "total_characters": self.total_characters,
            "total_tokens": self.total_tokens,
            "document_type": self.document_type,
            "file_size": self.file_size,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_modified": self.last_modified.isoformat()
            if self.last_modified
            else None,
            "outline": self.outline,
            "key_sections": self.key_sections,
        }


def create_vespa_schema_definition() -> dict[str, Any]:
    return {
        "schema": "policy_document",
        "document": {
            "fields": [
                {"name": "id", "type": "string", "indexing": "summary | attribute"},
                {"name": "title", "type": "string", "indexing": "index | summary"},
                {"name": "text", "type": "string", "indexing": "index | summary"},
                {
                    "name": "category",
                    "type": "string",
                    "indexing": "attribute | summary",
                },
                {
                    "name": "chunk_index",
                    "type": "int",
                    "indexing": "attribute | summary",
                },
                {
                    "name": "source_file",
                    "type": "string",
                    "indexing": "attribute | summary",
                },
                {
                    "name": "page_numbers",
                    "type": "array<int>",
                    "indexing": "attribute | summary",
                },
                {
                    "name": "page_range",
                    "type": "string",
                    "indexing": "attribute | summary",
                },
                {
                    "name": "headings",
                    "type": "array<string>",
                    "indexing": "attribute | summary",
                },
                {
                    "name": "char_count",
                    "type": "int",
                    "indexing": "attribute | summary",
                },
                {
                    "name": "token_count",
                    "type": "int",
                    "indexing": "attribute | summary",
                },
                {
                    "name": "document_id",
                    "type": "string",
                    "indexing": "attribute | summary",
                },
                {
                    "name": "previous_chunk_id",
                    "type": "string",
                    "indexing": "attribute | summary",
                },
                {
                    "name": "next_chunk_id",
                    "type": "string",
                    "indexing": "attribute | summary",
                },
                {
                    "name": "chunk_position",
                    "type": "float",
                    "indexing": "attribute | summary",
                },
                {
                    "name": "section_path",
                    "type": "array<string>",
                    "indexing": "attribute | summary",
                },
                {
                    "name": "importance_score",
                    "type": "float",
                    "indexing": "attribute | summary",
                },
            ]
        },
        "fieldsets": {"default": ["title", "text"]},
        "rank-profiles": {
            "default": {
                "first-phase": {
                    "expression": "nativeRank(title, text) + 0.1 * attribute(importance_score)"
                }
            },
            "with_position": {
                "first-phase": {
                    "expression": "nativeRank(title, text) * (1.0 - 0.3 * attribute(chunk_position))"
                }
            },
        },
    }


def create_document_metadata_schema() -> dict[str, Any]:
    return {
        "schema": "document_metadata",
        "document": {
            "fields": [
                {"name": "id", "type": "string", "indexing": "summary | attribute"},
                {
                    "name": "file_path",
                    "type": "string",
                    "indexing": "attribute | summary",
                },
                {"name": "file_name", "type": "string", "indexing": "index | summary"},
                {
                    "name": "category",
                    "type": "string",
                    "indexing": "attribute | summary",
                },
                {
                    "name": "total_pages",
                    "type": "int",
                    "indexing": "attribute | summary",
                },
                {
                    "name": "total_chunks",
                    "type": "int",
                    "indexing": "attribute | summary",
                },
                {
                    "name": "total_characters",
                    "type": "long",
                    "indexing": "attribute | summary",
                },
                {
                    "name": "total_tokens",
                    "type": "long",
                    "indexing": "attribute | summary",
                },
                {
                    "name": "document_type",
                    "type": "string",
                    "indexing": "attribute | summary",
                },
                {
                    "name": "file_size",
                    "type": "long",
                    "indexing": "attribute | summary",
                },
                {
                    "name": "created_at",
                    "type": "string",
                    "indexing": "attribute | summary",
                },
                {
                    "name": "last_modified",
                    "type": "string",
                    "indexing": "attribute | summary",
                },
                {
                    "name": "outline",
                    "type": "string",
                    "indexing": "summary",
                },
                {
                    "name": "key_sections",
                    "type": "array<string>",
                    "indexing": "attribute | summary",
                },
            ]
        },
    }
