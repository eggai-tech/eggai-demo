from libraries.core.config import BaseAgentConfig


class Settings(BaseAgentConfig):
    app_name: str = "test_agent"
    prometheus_metrics_port: int = 9999


def test_tracing_disabled_clears_endpoint(monkeypatch):
    monkeypatch.setenv("TRACING_ENABLED", "false")
    assert Settings().otel_endpoint == ""


def test_tracing_enabled_keeps_endpoint(monkeypatch):
    monkeypatch.setenv("OTEL_ENDPOINT", "http://collector:4318")
    assert Settings().otel_endpoint == "http://collector:4318"
