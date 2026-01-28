from opentelemetry import trace
from prometheus_client import start_http_server

from .metrics_definitions import (
    gen_ai_client_cost_current,
    gen_ai_client_cost_per_token,
    gen_ai_client_cost_total,
    gen_ai_client_operation_duration,
    gen_ai_client_token_usage,
    gen_ai_server_request_duration,
    gen_ai_server_time_to_first_token,
)
from .metrics_exporters import export_semantic_metrics, export_token_metrics
from .metrics_normalizers import normalize_gen_ai_system, normalize_operation_name
from .pricing import calculate_request_cost

# Re-export all public names for backward compatibility
__all__ = [
    "gen_ai_client_token_usage",
    "gen_ai_client_operation_duration",
    "gen_ai_server_request_duration",
    "gen_ai_server_time_to_first_token",
    "gen_ai_client_cost_total",
    "gen_ai_client_cost_current",
    "gen_ai_client_cost_per_token",
    "normalize_gen_ai_system",
    "normalize_operation_name",
    "export_semantic_metrics",
    "export_token_metrics",
    "calculate_request_cost",
    "patch_tracking_lm",
    "start_metrics_server",
    "init_token_metrics",
]


def patch_tracking_lm():
    from libraries.ml.dspy.language_model import TrackingLM

    original_forward = TrackingLM.forward

    def patched_forward(self, prompt=None, messages=None, **kwargs):
        import time

        start_time = time.time()

        result = original_forward(self, prompt, messages, **kwargs)

        operation_duration = time.time() - start_time

        try:
            current_span = trace.get_current_span()
            export_token_metrics(self, current_span)
            export_semantic_metrics(self, operation_duration, current_span)
        except Exception as e:
            print(f"Error exporting token metrics: {e}")

        return result

    TrackingLM.forward = patched_forward
    print("✓ Patched TrackingLM to export OpenTelemetry semantic convention metrics")


_metrics_server_started = False

def start_metrics_server(port: int = 9091):
    global _metrics_server_started
    if _metrics_server_started:
        print(f"⚠️  Metrics server already running, skipping start on port {port}")
        return
    try:
        start_http_server(port)
        _metrics_server_started = True
        print(f"✓ Prometheus metrics server started on port {port}")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"⚠️  Port {port} already in use, metrics server may be running in another process")
            _metrics_server_started = True
        else:
            raise


def init_token_metrics(port: int = 9091, application_name: str = "unknown", force_init: bool = False):
    global _application_name
    import os

    if os.getenv("CI") and not force_init:
        print("⚠️  Metrics collection disabled in CI environment")
        return

    from . import metrics_definitions

    metrics_definitions._application_name = application_name
    _application_name = application_name

    print("Initializing GenAI token metrics system...")
    print(f"Application name: {application_name}")

    patch_tracking_lm()
    start_metrics_server(port)

    print(f"✅ Token metrics system initialized on port {port}")
    print("✅ OpenTelemetry semantic convention metrics enabled")


try:
    patch_tracking_lm()
    print("✓ Auto-initialized TrackingLM patching for metrics export")
except Exception as e:
    print(f"Warning: Could not auto-initialize metrics patching: {e}")


if __name__ == "__main__":
    init_token_metrics()
    print("OpenTelemetry GenAI semantic convention metrics exporter is running...")

    import time

    while True:
        time.sleep(60)
