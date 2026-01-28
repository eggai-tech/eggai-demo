from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ClassifierV7Settings(BaseSettings):
    n_classes: int = Field(default=5)
    model_config = SettingsConfigDict(
        env_prefix="TRIAGE_V7_",
        env_file=".env",
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore',
        protected_namespaces=('settings_',)
    )

    model_name: str = Field(default="google/gemma-3-1b-it")
    use_qat_model: bool = Field(default=False)
    use_4bit: bool = Field(default=False)

    learning_rate: float = Field(default=2e-4)
    num_epochs: int = Field(default=10)
    batch_size: int = Field(default=1)
    gradient_accumulation_steps: int = Field(default=4)
    max_length: int = Field(default=512)

    lora_r: int = Field(default=16)
    lora_alpha: int = Field(default=16)
    lora_dropout: float = Field(default=0.05)

    output_dir: str = Field(default="./models/gemma3-triage-v7")

    def get_model_name(self) -> str:
        if self.use_qat_model:
            qat_mapping = {
                "google/gemma-3-1b-it": "google/gemma-3-qat-1b-it",
                "google/gemma-3-2b-it": "google/gemma-3-qat-2b-it",
                "google/gemma-3-9b-it": "google/gemma-3-qat-9b-it",
                "google/gemma-3-27b-it": "google/gemma-3-qat-27b-it",
                "google/gemma-2-2b-it": "google/gemma-2-2b-it",
                "google/gemma-2-9b-it": "google/gemma-2-9b-it",
                "google/gemma-2-27b-it": "google/gemma-2-27b-it",
            }
            return qat_mapping.get(self.model_name, self.model_name)
        return self.model_name

