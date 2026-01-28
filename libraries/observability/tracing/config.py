from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    service_namespace: str = Field(default="eggai")
    otel_endpoint: str = Field(default="http://localhost:4318")
    otel_endpoint_insecure: bool = Field(default=True)
    otel_exporter_otlp_protocol: str = Field(default="http/protobuf")
    tracing_enabled: bool = Field(default=True)

    disabled_instrumentors: list[str] = Field(default=["langchain"])
    sampling_rate: float = Field(default=1.0)

    model_config = SettingsConfigDict(
        env_prefix="TRACING_", env_file=".env", env_ignore_empty=True, extra="ignore"
    )


settings = Settings()
