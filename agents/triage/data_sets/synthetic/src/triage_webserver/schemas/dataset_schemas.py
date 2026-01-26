from datetime import datetime

from pydantic import BaseModel


class DatasetBase(BaseModel):
    name: str
    description: str | None = None
    model: str | None = None


class DatasetCreate(DatasetBase):
    total_target: int = 100
    agent_distribution: dict[str, float] | None = None
    special_case_distribution: dict[str, float] | None = None
    temperatures: list[float] = [0.7, 0.8, 0.9]
    turns: list[int] = [1, 3, 5]


class DatasetResponse(DatasetBase):
    id: int
    total_examples: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DatasetList(BaseModel):
    datasets: list[DatasetResponse]
    total: int
