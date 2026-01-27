import uuid

from eggai.schemas import Message
from pydantic import BaseModel, Field


# TODO make traceparent required and remove default
class TracedMessage(Message):
    traceparent: str | None = Field(
        default=None,
        description="W3C trace context traceparent header (version-traceID-spanID-flags)",
    )
    tracestate: str | None = Field(
        default=None,
        description="W3C trace context tracestate header with vendor-specific trace info",
    )
    service_tier: str | None = Field(
        default="standard",
        description="Service tier for gen_ai spans (standard, premium, etc.)",
    )


class GenAIAttributes(BaseModel):
    model_provider: str = Field(default="unknown")
    model_name: str = Field(default="unknown")
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    service_tier: str = Field(default="standard")
    token_count: int | None = None

    def to_span_attributes(self) -> dict[str, str]:
        result = {}
        for key, value in self.model_dump().items():
            if value is not None:
                # Ensure all values are strings if they're not already
                if not isinstance(value, str) and not isinstance(value, (list, bytes)):
                    value = str(value)
                result[f"gen_ai.{key}"] = value
        return result
