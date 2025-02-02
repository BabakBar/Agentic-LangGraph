"""Configure logging for the client application."""
import logging
import sys
import logging.config
from typing import Any, Dict

def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    log_config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "standard",
                "stream": sys.stdout,
            },
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["console"],
                "level": level,
                "propagate": True,
            },
            "client": {
                "handlers": ["console"],
                "level": level,
                "propagate": False,
            },
            "streamlit": {
                "handlers": ["console"],
                "level": level,
                "propagate": False,
            },
        },
    }
    
    logging.config.dictConfig(log_config)
    
    # Add startup message to verify logging
    logger = logging.getLogger(__name__)
    logger.info("Client logging configured successfully")