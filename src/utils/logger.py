import logging
import logging.config
from pathlib import Path

# Optional: create logs directory
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "pass_aoi.log"

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,  # let 3rd-party libs log too
    "formatters": {
        "standard": {"format": "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"},
        "detailed": {"format": "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "detailed",
            "level": "DEBUG",
            "filename": str(LOG_FILE),
            "encoding": "utf-8",
        },
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "DEBUG",  # global level
    },
}

logging.config.dictConfig(LOGGING_CONFIG)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with project-wide settings."""
    return logging.getLogger(name)
