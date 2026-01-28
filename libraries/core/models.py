from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str = Field(default="base_model")
    max_iterations: int = Field(default=5, ge=1, le=10)
    use_tracing: bool = Field(default=True)
    cache_enabled: bool = Field(default=False)
    timeout_seconds: float = Field(default=30.0, ge=1.0)
    truncation_length: int = Field(default=15000, ge=1000)

    model_config = {"validate_assignment": True}


class ModelResult(BaseModel):
    response: str = Field(...)
    processing_time_ms: int = Field(..., ge=0)
    success: bool = Field(default=True)
    truncated: bool = Field(default=False)
    original_length: int | None = Field(default=None)
    truncated_length: int | None = Field(default=None)
    error: str | None = Field(default=None)

    model_config = {"validate_assignment": True}
