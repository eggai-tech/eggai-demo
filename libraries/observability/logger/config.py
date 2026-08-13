from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    log_level: str = Field(default="INFO")
    # No env= overrides: that is a pydantic v1 argument which v2 silently files
    # under json_schema_extra, and it named the wrong variables anyway. With
    # env_prefix below, these read from LOGGER_LOG_FORMAT / LOGGER_SUPPRESS_LOGGERS.
    log_format: str = Field(
        default="%(asctime)s %(levelname)s %(name)s %(filename)s %(lineno)d %(funcName)s %(message)s",
    )

    log_formatter: str = Field(default="colored")
    suppress_loggers: list[str] = Field(
        default=["httpx", "urllib3", "asyncio", "aiokafka"]
    )

    suppress_level: str = Field(default="WARNING")

    model_config = SettingsConfigDict(
        env_prefix="LOGGER_", env_file=".env", env_ignore_empty=True, extra="ignore"
    )


settings = Settings()
