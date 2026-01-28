from enum import Enum


class ErrorCategory(Enum):
    USER_INPUT = "user_input"
    DATA_ACCESS = "data_access"
    SYSTEM = "system"
    MODEL = "model"
    SECURITY = "security"


class ErrorResponse:
    INVALID_CLAIM_NUMBER = "Please provide a valid claim number."
    CLAIM_NOT_FOUND = "I couldn't find a claim with that number. Please verify the claim number and try again."
    INVALID_POLICY_NUMBER = "Please provide a valid policy number."
    MISSING_CLAIM_DETAILS = "Please provide details about the incident."
    MISSING_FIELD_NAME = "Please specify which field you want to update."
    MISSING_FIELD_VALUE = "Please provide a value to update the field with."
    FIELD_NOT_FOUND = "That field doesn't exist in our claims records. Available fields include contact information, incident details, and claim status."
    INVALID_FIELD_VALUE = "The value provided isn't valid for this field. Please try again with a proper format."

    GENERIC_ERROR = "I apologize, but I'm experiencing a technical issue. Please try again in a moment."
    DATA_ERROR = "I'm having trouble accessing the claims database right now. Please try again shortly."
    MODEL_ERROR = "I'm having difficulty processing your request right now. Please try again or rephrase your question."

    SECURITY_FIELD_BLOCKED = "For security reasons, that field cannot be updated. Please contact customer service for assistance."
    SECURITY_INPUT_BLOCKED = "For security reasons, I cannot process that information. Please avoid including sensitive data like credit card numbers or social security numbers."


def get_user_friendly_error(
    error: Exception,
    category: ErrorCategory = ErrorCategory.SYSTEM,
    fallback_message: str | None = None,
) -> str:
    error_str = str(error).lower()

    if fallback_message:
        return fallback_message

    if category == ErrorCategory.USER_INPUT:
        if "claim" in error_str and "not found" in error_str:
            return ErrorResponse.CLAIM_NOT_FOUND
        elif "invalid" in error_str and "claim" in error_str:
            return ErrorResponse.INVALID_CLAIM_NUMBER
        elif "invalid" in error_str and "policy" in error_str:
            return ErrorResponse.INVALID_POLICY_NUMBER
        elif "field" in error_str and "cannot" in error_str:
            return ErrorResponse.SECURITY_FIELD_BLOCKED
        else:
            return f"There was an issue with your input: {error}"

    elif category == ErrorCategory.SECURITY:
        return ErrorResponse.SECURITY_FIELD_BLOCKED

    elif category == ErrorCategory.MODEL:
        return ErrorResponse.MODEL_ERROR

    elif category == ErrorCategory.DATA_ACCESS:
        return ErrorResponse.DATA_ERROR

    return ErrorResponse.GENERIC_ERROR
