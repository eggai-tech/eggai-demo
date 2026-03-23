from libraries.observability.logger import get_console_logger

logger = get_console_logger("frontend_agent")

try:
    from guardrails import AsyncGuard
except Exception as e:
    logger.warning(f"Guardrails import failed; disabling guardrails: {e}")
    AsyncGuard = None

try:
    from guardrails.hub import ToxicLanguage
except Exception as e:
    logger.warning(f"ToxicLanguage validator unavailable; disabling guardrails: {e}")
    ToxicLanguage = None

_toxic_language_guard = None
if AsyncGuard and ToxicLanguage:
    _toxic_language_guard = AsyncGuard().use(
        ToxicLanguage,
        threshold=0.5,
        validation_method="sentence",
        on_fail="noop",
    )


async def toxic_language_guard(text: str) -> str | None:
    if _toxic_language_guard is None:
        return text
    result = await _toxic_language_guard.validate(text)
    if result.validation_passed is False:
        return None
    return result.validated_output
