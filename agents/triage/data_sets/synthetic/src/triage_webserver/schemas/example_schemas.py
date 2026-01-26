from datetime import datetime

from pydantic import BaseModel


class ExampleBase(BaseModel):
    conversation: str
    target_agent: str
    turns: int
    temperature: float
    special_case: str | None = None


class ExampleCreate(ExampleBase):
    dataset_id: int
    index_batch: int | None = None
    total_batch: int | None = None


class ExampleUpdate(BaseModel):
    conversation: str | None = None
    target_agent: str | None = None
    turns: int | None = None
    temperature: float | None = None
    special_case: str | None = None


class ExampleResponse(ExampleBase):
    id: int
    dataset_id: int
    index_batch: int | None
    total_batch: int | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ExampleList(BaseModel):
    examples: list[ExampleResponse]
    total: int
