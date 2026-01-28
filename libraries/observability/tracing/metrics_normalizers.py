def normalize_gen_ai_system(model_name: str) -> str:
    model_lower = model_name.lower()

    if any(
        x in model_lower
        for x in ["gpt", "openai", "o1-", "davinci", "curie", "babbage", "ada"]
    ):
        return "openai"

    elif any(x in model_lower for x in ["claude", "anthropic"]):
        return "anthropic"

    elif "bedrock" in model_lower:
        return "aws.bedrock"

    elif "azure" in model_lower and "inference" in model_lower:
        return "az.ai.inference"

    elif "azure" in model_lower and "openai" in model_lower:
        return "az.ai.openai"

    elif "cohere" in model_lower:
        return "cohere"

    elif "deepseek" in model_lower:
        return "deepseek"

    elif any(x in model_lower for x in ["gemini", "generativelanguage"]):
        return "gcp.gemini"

    elif any(x in model_lower for x in ["vertex", "aiplatform"]):
        return "gcp.vertex_ai"

    elif any(x in model_lower for x in ["google", "palm", "bard"]):
        return "gcp.gen_ai"

    elif "groq" in model_lower:
        return "groq"

    elif any(x in model_lower for x in ["watsonx", "ibm"]):
        return "ibm.watsonx.ai"

    elif any(x in model_lower for x in ["llama", "meta"]):
        return "meta"

    elif "mistral" in model_lower:
        return "mistral_ai"

    elif "perplexity" in model_lower:
        return "perplexity"

    elif "xai" in model_lower:
        return "xai"

    elif any(
        x in model_lower for x in ["lm_studio", "lm-studio", "local", "localhost"]
    ):
        return "_OTHER"

    else:
        return "_OTHER"


def normalize_operation_name(
    model_name: str, prompt: str = None, messages: list = None
) -> str:
    if messages:
        return "chat"

    model_lower = model_name.lower()
    if any(x in model_lower for x in ["embedding", "embed", "ada-002"]):
        return "embeddings"

    if any(x in model_lower for x in ["gemini", "generate-content"]):
        return "generate_content"

    if any(x in model_lower for x in ["gpt-4", "gpt-3.5", "claude", "gemini", "llama"]):
        return "chat"
    elif any(x in model_lower for x in ["davinci", "curie", "babbage", "ada"]):
        return "text_completion"

    return "chat"
