from opentelemetry import trace

from libraries.observability.tracing.otel import init_telemetry


def test_empty_endpoint_installs_nothing(monkeypatch):
    installed = []
    monkeypatch.setattr(trace, "set_tracer_provider", installed.append)
    init_telemetry("no_tracing_agent", endpoint="")
    assert installed == []
