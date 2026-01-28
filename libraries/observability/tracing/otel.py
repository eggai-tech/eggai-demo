import asyncio
import functools
import json
import os
import random
import uuid
from asyncio import iscoroutine
from collections.abc import Awaitable, Callable
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import SpanLimits
from opentelemetry.trace import SpanContext, TraceFlags, Tracer, TraceState

from libraries.observability.logger import get_console_logger

logger = get_console_logger("tracing.dspy")


def safe_set_attribute(span, key: str, value: Any) -> None:
    if value is None:
        return

    # gen_ai attributes must be strings per OTel semantic conventions
    if key.startswith("gen_ai."):
        if isinstance(value, (bool, int, float)):
            value = str(value)
        elif not isinstance(value, (str, bytes)) and not isinstance(value, list):
            try:
                value = str(value)
            except Exception:
                logger.debug(
                    f"Skipping gen_ai attribute {key} with unconvertible type: {type(value)}"
                )
                return

    if isinstance(value, (bool, int, float, str, bytes)):
        try:
            span.set_attribute(key, value)
        except Exception as e:
            logger.debug(f"Error setting span attribute {key}: {e}")
            try:
                span.set_attribute(key, str(value))
            except Exception:
                logger.debug(
                    f"Failed to set attribute {key} even after string conversion"
                )

    elif isinstance(value, list) and all(
        isinstance(item, (bool, int, float, str, bytes)) for item in value
    ):
        try:
            span.set_attribute(key, value)
        except Exception:
            try:
                span.set_attribute(key, str(value))
            except Exception:
                logger.debug(f"Failed to set list attribute {key}")

    else:
        try:
            span.set_attribute(key, str(value))
        except Exception:
            logger.debug(
                f"Skipping attribute {key} with invalid value type: {type(value)}"
            )


def init_telemetry(app_name: str, endpoint: str | None = None) -> None:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    otlp_endpoint = endpoint or os.getenv("OTEL_ENDPOINT", "http://localhost:4318")

    resource = Resource.create({"service.name": app_name})

    limits = SpanLimits(max_span_attribute_length=32768)

    trace.set_tracer_provider(TracerProvider(resource=resource, span_limits=limits))
    otlp_exporter = OTLPSpanExporter(endpoint=f"{otlp_endpoint}/v1/traces")
    span_processor = BatchSpanProcessor(otlp_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)

    # Patch Span.set_attribute for safer attribute handling
    try:
        logger.info(
            "Attempting to patch span attribute setter for safer attribute handling"
        )
        from opentelemetry.sdk.trace import Span as SDKSpan
        from opentelemetry.trace import Span

        span_classes = []
        if hasattr(Span, "set_attribute"):
            span_classes.append(Span)
        if hasattr(SDKSpan, "set_attribute"):
            span_classes.append(SDKSpan)

        for span_class in span_classes:
            if hasattr(span_class, "set_attribute"):
                logger.info(
                    f"Patching {span_class.__name__}.set_attribute for safe attribute handling"
                )
                original_set_attribute = span_class.set_attribute

                def create_safe_middleware(original_method):
                    def safe_set_attribute_middleware(self, key: str, value: Any):
                        if value is None:
                            return self

                        if key.startswith("gen_ai."):
                            if isinstance(value, (bool, int, float)):
                                value = str(value)
                            elif not isinstance(value, (str, bytes)):
                                try:
                                    value = str(value)
                                except Exception:
                                    return self

                        try:
                            return original_method(self, key, value)
                        except Exception:
                            try:
                                return original_method(self, key, str(value))
                            except Exception:
                                return self

                    return safe_set_attribute_middleware

                span_class.set_attribute = create_safe_middleware(
                    original_set_attribute
                )
                logger.info(f"Successfully patched {span_class.__name__}.set_attribute")
    except Exception as e:
        logger.warning(
            f"Unable to patch span attribute setter: {e}. Using manual safe_set_attribute instead."
        )


_TRACERS: dict[str, Tracer] = {}


def get_tracer(name: str) -> Tracer:
    if name not in _TRACERS:
        _TRACERS[name] = trace.get_tracer(name)
    return _TRACERS[name]


def _normalize_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def create_tracer(name: str, component: str | None = None) -> Tracer:
    normalized = _normalize_name(name)
    tracer_name = (
        f"{normalized}.{_normalize_name(component)}" if component else normalized
    )
    tracer = get_tracer(tracer_name)
    tracer.name = tracer_name
    return tracer


def extract_span_context(
    traceparent: str, tracestate: str | None = None
) -> SpanContext | None:
    parts = traceparent.split("-")
    if len(parts) != 4 or parts[0] != "00":
        return None
    try:
        trace_id = int(parts[1], 16)
        span_id = int(parts[2], 16)
        trace_flags = TraceFlags(int(parts[3], 16))
    except Exception:
        return None
    state = TraceState.from_header(tracestate) if tracestate else TraceState()
    return SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=True,
        trace_flags=trace_flags,
        trace_state=state,
    )


def format_span_as_traceparent(span) -> tuple:
    sc = span.get_span_context()
    traceparent = f"00-{sc.trace_id:032x}-{sc.span_id:016x}-{int(sc.trace_flags):02x}"
    tracestate = str(sc.trace_state) if sc.trace_state else ""
    return traceparent, tracestate


def traced_handler(span_name: str = None):
    def decorator(handler_func: Callable[[dict], Awaitable[None]]):
        @functools.wraps(handler_func)
        async def wrapper(*args, **kwargs):
            from libraries.observability.tracing.schemas import TracedMessage

            splitted = handler_func.__module__.split(".")
            module_name = splitted[-2] if len(splitted) > 1 else splitted[0]
            tracer_name = f"{module_name}_agent"
            tracer = get_tracer(tracer_name)

            msg = next((arg for arg in args if isinstance(arg, TracedMessage)), None)
            if not msg:
                msg = next(
                    (v for v in kwargs.values() if isinstance(v, TracedMessage)), None
                )
            if not msg:
                raw = next(
                    (arg for arg in args if isinstance(arg, dict)), None
                ) or next((v for v in kwargs.values() if isinstance(v, dict)), None)
                if raw:
                    msg = TracedMessage(**raw)

            if isinstance(msg, dict) and isinstance(msg.get("channel"), dict):
                original = msg["channel"]
                msg["channel"] = original.get("channel") or json.dumps(original)

            parent_context = None
            traceparent = getattr(msg, "traceparent", None)
            tracestate = getattr(msg, "tracestate", None)
            if traceparent:
                span_ctx = extract_span_context(traceparent, tracestate)
                if span_ctx:
                    parent_context = trace.set_span_in_context(
                        trace.NonRecordingSpan(span_ctx)
                    )

            span_name_to_use = span_name or f"handle_{module_name}_message"
            with tracer.start_as_current_span(
                span_name_to_use, context=parent_context, kind=trace.SpanKind.SERVER
            ) as span:
                safe_set_attribute(span, "agent.name", module_name)
                safe_set_attribute(span, "agent.handler", handler_func.__name__)
                safe_set_attribute(
                    span, "message.id", str(getattr(msg, "id", "unknown"))
                )

                if iscoroutine(handler_func) or asyncio.iscoroutinefunction(
                    handler_func
                ):
                    return await handler_func(*args, **kwargs)
                else:
                    return handler_func(*args, **kwargs)

        return wrapper

    return decorator


def get_traceparent_from_connection_id(connection_id: str) -> str:
    connection_uuid = uuid.UUID(connection_id)
    trace_id = connection_uuid.hex
    span_id = f"{random.getrandbits(64):016x}"
    trace_flags = "01"
    return f"00-{trace_id}-{span_id}-{trace_flags}"
