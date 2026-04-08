import re

# Pattern to detect field:value queries (e.g. document_id:"auto_policy", source_file:auto.md)
FIELD_QUERY_PATTERN = re.compile(r'^(\w+):\s*"?([^"]+)"?\s*$')

# Only support explicit field filtering for fields we store and query today.
SUPPORTED_FIELD_QUERY_FIELDS = {
    "id",
    "document_id",
    "source_file",
    "category",
}


def parse_field_query(query: str) -> tuple[str, str] | None:
    """Return a supported field query as ``(field, value)`` if present."""
    match = FIELD_QUERY_PATTERN.match(query)
    if not match:
        return None

    field, value = match.group(1), match.group(2)
    if field not in SUPPORTED_FIELD_QUERY_FIELDS:
        return None

    return field, value


def escape_yql_string(value: str) -> str:
    """Escape a string for safe inclusion in a quoted YQL literal."""
    return value.replace("\\", "\\\\").replace('"', '\\"')
