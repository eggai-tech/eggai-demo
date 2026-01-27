from typing import Literal, get_args

from pydantic import Field

from libraries.communication.protocol import (
    ChatMessage as ChatMessage,
)
from libraries.core import (
    ModelConfig as BaseModelConfig,
)

PolicyCategory = Literal["auto", "life", "home", "health"]
VALID_CATEGORIES: frozenset[str] = frozenset(get_args(PolicyCategory))


class ModelConfig(BaseModelConfig):
    name: str = Field("policies_react", description="Name of the model")
    date_format: str = Field("YYYY-MM-DD", description="Required date format for responses")
