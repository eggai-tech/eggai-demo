from typing import Literal, TypedDict

from libraries.communication.protocol import (
    MessageData,
)
from libraries.core import (
    ModelConfig as BaseModelConfig,
)

WebSocketStateType = Literal["connected", "disconnected", "connecting"]


class UserMessage(TypedDict):
    id: str
    type: Literal["user_message"]
    source: Literal["Frontend"]
    data: MessageData
    traceparent: str | None
    tracestate: str | None


class AgentResponseMessage(TypedDict):
    id: str
    type: Literal["agent_message"]
    source: str
    data: MessageData
    traceparent: str | None
    tracestate: str | None


class ModelConfig(BaseModelConfig):
    pass
