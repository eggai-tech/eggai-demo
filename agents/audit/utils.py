from uuid import uuid4

from libraries.observability.logger import get_console_logger
from libraries.observability.tracing import TracedMessage

logger = get_console_logger("audit_agent")

def get_message_metadata(
    message: TracedMessage | dict | None
) -> tuple[str, str]:
    if message is None:
        return "unknown", "unknown"

    mtype = getattr(message, "type", None)
    msource = getattr(message, "source", None)
    if mtype is not None or msource is not None:
        return mtype or "unknown", msource or "unknown"

    try:
        return message.get("type", "unknown"), message.get("source", "unknown")
    except Exception:
        logger.warning("Could not extract message type/source from %r", message)
        return "unknown", "unknown"


def get_message_content(message: TracedMessage | dict | None) -> str | None:
    data = getattr(message, "data", None)
    if not isinstance(data, dict):
        return None

    content = data.get("message")
    if isinstance(content, str):
        return content

    chat = data.get("chat_messages")
    if isinstance(chat, list) and chat:
        last = chat[-1]
        if isinstance(last, dict):
            return last.get("content")

    return None


def get_message_id(message: TracedMessage | dict | None) -> str:
    if message is None:
        return f"null_message_{uuid4()}"

    mid = getattr(message, "id", None)
    return str(mid) if mid is not None else str(uuid4())


def propagate_trace_context(
    source_message: TracedMessage | dict | None, target_message: TracedMessage
) -> None:
    if source_message is None:
        return

    if hasattr(source_message, "traceparent") and source_message.traceparent:
        target_message.traceparent = source_message.traceparent
    if hasattr(source_message, "tracestate") and source_message.tracestate:
        target_message.tracestate = source_message.tracestate
