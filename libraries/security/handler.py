import jwt
from eggai import Channel

from libraries.communication.streaming import publish_error_message
from libraries.observability.logger import get_console_logger
from libraries.observability.tracing import TracedMessage

from .gateway import Run, run_var
from .keycloak import Keycloak, caller_var

logger = get_console_logger("security.handler")


async def authenticate(
    keycloak: Keycloak, msg: TracedMessage, human_channel: Channel, agent_name: str, connection_id: str
) -> bool:
    caller_var.set(None)
    token = (msg.data.get("security_context") or {}).get("access_token", "")
    run_var.set(Run(token, msg.traceparent or "", msg.data["message_id"]))
    if not keycloak.enabled:
        return True
    try:
        caller_var.set(keycloak.validate(token))
    except jwt.PyJWTError as e:
        logger.warning(f"Authentication failed for {connection_id}: {e}")
        await publish_error_message(
            human_channel, agent_name, connection_id,
            message=f"Authentication failed: {e}",
            traceparent=msg.traceparent, tracestate=msg.tracestate,
        )
        return False
    return True
