from typing import Any

from libraries.observability.logger import get_console_logger

logger = get_console_logger("libraries.patches")


def patch_usage_tracker():
    # FIXME: UsageTracker doesn't handle nested dicts in usage entries
    def _merge_usage_entries(
        self: Any,
        usage_entry1: dict[str, Any] | None,
        usage_entry2: dict[str, Any] | None,
    ) -> dict[str, dict[str, Any]]:
        if not usage_entry1:
            return dict(usage_entry2 or {})
        if not usage_entry2:
            return dict(usage_entry1 or {})

        result = dict(usage_entry2)

        for key, val1 in usage_entry1.items():
            val2 = result.get(key)

            if isinstance(val1, dict) or isinstance(val2, dict):
                sub1 = val1 if isinstance(val1, dict) else {}
                sub2 = val2 if isinstance(val2, dict) else {}
                result[key] = self._merge_usage_entries(sub1, sub2)

            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                result[key] = val1 + val2

            else:
                # Non-numeric usage fields (e.g. a model/service-tier label)
                # can't be summed - keep whichever side actually has one.
                result[key] = val2 if val2 is not None else val1

        return result

    from dspy.utils.usage_tracker import UsageTracker

    UsageTracker._merge_usage_entries = _merge_usage_entries
    logger.info("DSPY UsageTracker patched successfully")
