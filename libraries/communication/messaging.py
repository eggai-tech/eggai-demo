from collections.abc import Callable
from typing import (
    Any,
    Literal,
    Protocol,
    TypedDict,
    TypeVar,
)

from eggai import Agent, Channel
from faststream.kafka import KafkaMessage

from libraries.communication.protocol import (
    AgentName,
    BillingRequestMessage,
    ClaimRequestMessage,
    EscalationRequestMessage,
    MessageType,
    OffsetReset,
    PolicyRequestMessage,
    UserMessage,
)
from libraries.observability.tracing import TracedMessage

T = TypeVar('T', bound=dict[str, Any])


class MessageFilter(Protocol[T]):
    def __call__(self, msg: T) -> bool: ...


def create_type_filter(message_type: str | MessageType) -> MessageFilter[dict[str, Any]]:
    return lambda msg: msg.get("type") == message_type


def create_source_filter(source: str | AgentName) -> MessageFilter[dict[str, Any]]:
    return lambda msg: msg.get("source") == source


def create_compound_filter(
    message_type: str | MessageType | None = None,
    source: str | AgentName | None = None,
) -> MessageFilter[dict[str, Any]]:
    def filter_func(msg: dict[str, Any]) -> bool:
        if message_type and msg.get("type") != message_type:
            return False
        if source and msg.get("source") != source:
            return False
        return True
    return filter_func


HandlerT = TypeVar('HandlerT', bound=Callable)


class SubscribeConfig(TypedDict, total=False):
    filter_by_message: MessageFilter
    group_id: str
    auto_offset_reset: OffsetReset
    enable_auto_commit: bool
    max_poll_records: int
    max_poll_interval_ms: int


def subscribe(
    agent: Agent,
    channel: Channel,
    *,
    message_type: str | MessageType | None = None,
    source: str | AgentName | None = None,
    filter_func: MessageFilter | None = None,
    group_id: str | None = None,
    auto_offset_reset: OffsetReset | Literal["latest", "earliest", "none"] = OffsetReset.LATEST,
    **kwargs: Any
) -> Callable[[HandlerT], HandlerT]:
    if not filter_func:
        if message_type or source:
            filter_func = create_compound_filter(message_type, source)

    # Build subscribe kwargs
    subscribe_kwargs: dict[str, Any] = {
        "auto_offset_reset": auto_offset_reset,
        **kwargs
    }

    if filter_func:
        subscribe_kwargs["filter_by_message"] = filter_func

    if group_id:
        subscribe_kwargs["group_id"] = group_id

    return agent.subscribe(channel=channel, **subscribe_kwargs)


typed_subscribe = subscribe

__all__ = [
    "create_type_filter",
    "create_source_filter",
    "create_compound_filter",
    "subscribe",
    "typed_subscribe",
    "BillingHandler",
    "ClaimsHandler",
    "PolicyHandler",
    "EscalationHandler",
    "UserMessageHandler",
    "AuditHandler",
]

BillingHandler = Callable[[BillingRequestMessage | TracedMessage, KafkaMessage], None]
ClaimsHandler = Callable[[ClaimRequestMessage | TracedMessage, KafkaMessage], None]
PolicyHandler = Callable[[PolicyRequestMessage | TracedMessage, KafkaMessage], None]
EscalationHandler = Callable[[EscalationRequestMessage | TracedMessage, KafkaMessage], None]
UserMessageHandler = Callable[[UserMessage | TracedMessage, KafkaMessage], None]
AuditHandler = Callable[[TracedMessage | dict, KafkaMessage], TracedMessage | dict | None]


