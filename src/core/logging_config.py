"""Configure logging for the application."""
import logging
import sys
import logging.config
import uuid
from pythonjsonlogger import jsonlogger
from typing import Any, Dict

def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    log_config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(timestamp)s %(level)s %(name)s %(message)s %(correlation_id)s",
                "rename_fields": {
                    "levelname": "level",
                    "asctime": "timestamp"
                },
                "json_default": str,
                "timestamp": True
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
            "agents": {
                "handlers": ["console"],
                "level": level,
                "propagate": False,
            },
            "service": {
                "handlers": ["console"],
                "level": level,
                "propagate": False,
            },
            "uvicorn": {
                "handlers": ["console"],
                "level": level,
                "propagate": False,
            },
        },
    }

    # Add correlation ID filter to all handlers
    correlation_id = str(uuid.uuid4())
    for handler in log_config["handlers"].values():
        handler["filters"] = ["correlation"]
    log_config["filters"] = {"correlation": {"()": "core.logging_config.CorrelationFilter", "correlation_id": correlation_id}}
    
    logging.config.dictConfig(log_config)
    
    # Add startup message to verify logging
    logger = logging.getLogger(__name__)
    logger.info("Logging configured successfully", extra={"correlation_id": correlation_id})

class CorrelationFilter(logging.Filter):
    """Add correlation ID to log records."""
    def __init__(self, correlation_id: str):
        super().__init__()
        self.correlation_id = correlation_id

    def filter(self, record):
        record.correlation_id = self.correlation_id
        return True