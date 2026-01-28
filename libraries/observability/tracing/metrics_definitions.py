from prometheus_client import Counter, Gauge, Histogram

gen_ai_client_token_usage = Counter(
    "gen_ai_client_token_usage",
    "Number of tokens used in GenAI requests",
    [
        "gen_ai_system",
        "gen_ai_request_model",
        "gen_ai_operation_name",
        "gen_ai_token_type",
        "gen_ai_agent_name",
    ],
)

gen_ai_client_operation_duration = Histogram(
    "gen_ai_client_operation_duration",
    "GenAI operation duration",
    [
        "gen_ai_system",
        "gen_ai_request_model",
        "gen_ai_operation_name",
        "gen_ai_agent_name",
    ],
    buckets=[
        0.01,
        0.02,
        0.04,
        0.08,
        0.16,
        0.32,
        0.64,
        1.28,
        2.56,
        5.12,
        10.24,
        20.48,
        40.96,
        81.92,
    ],
)

gen_ai_server_request_duration = Histogram(
    "gen_ai_server_request_duration",
    "GenAI server request duration",
    [
        "gen_ai_system",
        "gen_ai_request_model",
        "gen_ai_operation_name",
        "gen_ai_agent_name",
    ],
    buckets=[
        0.01,
        0.02,
        0.04,
        0.08,
        0.16,
        0.32,
        0.64,
        1.28,
        2.56,
        5.12,
        10.24,
        20.48,
        40.96,
        81.92,
    ],
)

gen_ai_server_time_to_first_token = Histogram(
    "gen_ai_server_time_to_first_token",
    "Time to generate first token for successful responses",
    [
        "gen_ai_system",
        "gen_ai_request_model",
        "gen_ai_operation_name",
        "gen_ai_agent_name",
    ],
    buckets=[
        0.001,
        0.005,
        0.01,
        0.02,
        0.04,
        0.06,
        0.08,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        2.5,
        5.0,
        7.5,
        10.0,
    ],
)

gen_ai_client_cost_total = Counter(
    "gen_ai_client_cost_total",
    "Total cost in USD for GenAI requests",
    [
        "gen_ai_system",
        "gen_ai_request_model",
        "gen_ai_operation_name",
        "cost_type",
        "gen_ai_agent_name",
    ],
)

gen_ai_client_cost_current = Gauge(
    "gen_ai_client_cost_current",
    "Cost in USD for the current request",
    [
        "gen_ai_system",
        "gen_ai_request_model",
        "gen_ai_operation_name",
        "cost_type",
        "gen_ai_agent_name",
    ],
)

gen_ai_client_cost_per_token = Gauge(
    "gen_ai_client_cost_per_token",
    "Cost per token for different models",
    [
        "gen_ai_system",
        "gen_ai_request_model",
        "gen_ai_operation_name",
        "gen_ai_token_type",
        "gen_ai_agent_name",
    ],
)

_application_name = "unknown"
