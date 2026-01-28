import logging
import logging.config
import sys
from typing import Any

from colorlog import ColoredFormatter

from libraries.observability.logger.config import settings

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": settings.log_format,
        },
        "standard": {
            "format": settings.log_format,
        },
        "colored": {
            "()": "colorlog.ColoredFormatter",
            "format": "%(log_color)s%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            "datefmt": "%H:%M:%S",
            "log_colors": {
                "DEBUG": "white",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        },
    },
    "handlers": {
        "default": {
            "level": settings.log_level,
            "formatter": settings.log_formatter,
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "": {
            "level": settings.log_level,
            "handlers": ["default"],
            "propagate": True,
        },
        "uvicorn.error": {
            "level": "INFO",
            "handlers": ["default"],
            "propagate": False,
        },
        "uvicorn.access": {
            "level": "INFO",
            "handlers": ["default"],
            "propagate": False,
        },
    },
}

for logger_name in settings.suppress_loggers:
    LOGGING_CONFIG["loggers"][logger_name] = {
        "level": settings.suppress_level,
        "handlers": ["default"],
        "propagate": False,
    }


def configure_logging(config: dict[str, Any] | None = None) -> None:
    logging_config = config or LOGGING_CONFIG
    logging.config.dictConfig(logging_config)


def get_logger(service_name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(service_name)


def get_console_logger(service_name: str) -> logging.Logger:
    logger = logging.getLogger(service_name)

    if logger.handlers:
        return logger

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    log_colors = {
        "DEBUG": "white",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    }

    console_formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%H:%M:%S",
        log_colors=log_colors,
    )

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(settings.log_level)
    console_handler.setFormatter(console_formatter)

    for logger_name in settings.suppress_loggers:
        logging.getLogger(logger_name).setLevel(settings.suppress_level)

    logger.setLevel(settings.log_level)
    logger.addHandler(console_handler)
    logger.propagate = False

    return logger
