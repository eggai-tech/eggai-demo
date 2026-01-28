from .metrics_definitions import (
    _application_name,
    gen_ai_client_cost_current,
    gen_ai_client_cost_per_token,
    gen_ai_client_cost_total,
    gen_ai_client_operation_duration,
    gen_ai_client_token_usage,
)
from .metrics_normalizers import normalize_gen_ai_system, normalize_operation_name


def export_semantic_metrics(lm, operation_duration: float = None, span=None):
    from . import init_metrics, metrics_definitions

    model_name = getattr(lm, "model_name", "unknown")
    gen_ai_system = normalize_gen_ai_system(model_name)
    gen_ai_operation_name = normalize_operation_name(model_name)

    agent_name = metrics_definitions._application_name

    cost_info = None
    if lm.prompt_tokens > 0 or lm.completion_tokens > 0:
        try:
            cost_info = init_metrics.calculate_request_cost(
                model_name, lm.prompt_tokens, lm.completion_tokens
            )
        except Exception as e:
            print(f"Error calculating cost: {e}")
            cost_info = None

    if lm.prompt_tokens > 0:
        gen_ai_client_token_usage.labels(
            **{
                "gen_ai_system": gen_ai_system,
                "gen_ai_request_model": model_name,
                "gen_ai_operation_name": gen_ai_operation_name,
                "gen_ai_token_type": "input",
                "gen_ai_agent_name": agent_name,
            }
        ).inc(lm.prompt_tokens)

    if lm.completion_tokens > 0:
        gen_ai_client_token_usage.labels(
            **{
                "gen_ai_system": gen_ai_system,
                "gen_ai_request_model": model_name,
                "gen_ai_operation_name": gen_ai_operation_name,
                "gen_ai_token_type": "output",
                "gen_ai_agent_name": agent_name,
            }
        ).inc(lm.completion_tokens)

    if operation_duration is not None:
        gen_ai_client_operation_duration.labels(
            **{
                "gen_ai_system": gen_ai_system,
                "gen_ai_request_model": model_name,
                "gen_ai_operation_name": gen_ai_operation_name,
                "gen_ai_agent_name": agent_name,
            }
        ).observe(operation_duration)

    if cost_info:
        gen_ai_client_cost_total.labels(
            **{
                "gen_ai_system": gen_ai_system,
                "gen_ai_request_model": model_name,
                "gen_ai_operation_name": gen_ai_operation_name,
                "cost_type": "input",
                "gen_ai_agent_name": agent_name,
            }
        ).inc(cost_info["prompt_cost"])

        gen_ai_client_cost_total.labels(
            **{
                "gen_ai_system": gen_ai_system,
                "gen_ai_request_model": model_name,
                "gen_ai_operation_name": gen_ai_operation_name,
                "cost_type": "output",
                "gen_ai_agent_name": agent_name,
            }
        ).inc(cost_info["completion_cost"])

        gen_ai_client_cost_total.labels(
            **{
                "gen_ai_system": gen_ai_system,
                "gen_ai_request_model": model_name,
                "gen_ai_operation_name": gen_ai_operation_name,
                "cost_type": "total",
                "gen_ai_agent_name": agent_name,
            }
        ).inc(cost_info["total_cost"])

        gen_ai_client_cost_current.labels(
            **{
                "gen_ai_system": gen_ai_system,
                "gen_ai_request_model": model_name,
                "gen_ai_operation_name": gen_ai_operation_name,
                "cost_type": "input",
                "gen_ai_agent_name": agent_name,
            }
        ).set(cost_info["prompt_cost"])

        gen_ai_client_cost_current.labels(
            **{
                "gen_ai_system": gen_ai_system,
                "gen_ai_request_model": model_name,
                "gen_ai_operation_name": gen_ai_operation_name,
                "cost_type": "output",
                "gen_ai_agent_name": agent_name,
            }
        ).set(cost_info["completion_cost"])

        gen_ai_client_cost_current.labels(
            **{
                "gen_ai_system": gen_ai_system,
                "gen_ai_request_model": model_name,
                "gen_ai_operation_name": gen_ai_operation_name,
                "cost_type": "total",
                "gen_ai_agent_name": agent_name,
            }
        ).set(cost_info["total_cost"])

        gen_ai_client_cost_per_token.labels(
            **{
                "gen_ai_system": gen_ai_system,
                "gen_ai_request_model": model_name,
                "gen_ai_operation_name": gen_ai_operation_name,
                "gen_ai_token_type": "input",
                "gen_ai_agent_name": agent_name,
            }
        ).set(cost_info["prompt_price_per_1k"] / 1000.0)

        gen_ai_client_cost_per_token.labels(
            **{
                "gen_ai_system": gen_ai_system,
                "gen_ai_request_model": model_name,
                "gen_ai_operation_name": gen_ai_operation_name,
                "gen_ai_token_type": "output",
                "gen_ai_agent_name": agent_name,
            }
        ).set(cost_info["completion_price_per_1k"] / 1000.0)


def export_token_metrics(lm, span=None):
    from . import init_metrics

    init_metrics.export_semantic_metrics(lm, span=span)
