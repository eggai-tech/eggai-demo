from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    log_level: str = Field(default="INFO")
    log_format: str = Field(
        default="%(asctime)s %(levelname)s %(name)s %(filename)s %(lineno)d %(funcName)s %(message)s",
        env="LOG_FORMAT",
    )

    log_formatter: str = Field(default="colored")
    suppress_loggers: list[str] = Field(
        default=["httpx", "urllib3", "asyncio", "aiokafka"], env="SUPPRESS_LOGGERS"
    )

    suppress_level: str = Field(default="WARNING")

    model_config = SettingsConfigDict(
        env_prefix="LOGGER_", env_file=".env", env_ignore_empty=True, extra="ignore"
    )


settings = Settings()
