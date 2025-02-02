"""Configure logging for the application."""
import logging
import sys
import logging.config
import uuid
from pythonjsonlogger import jsonlogger
from typing import Any, Dict
import os
import json

class StreamFormatter(logging.Formatter):
    """Custom formatter for stream events that cleans up the output."""
    
    def format(self, record):
        if hasattr(record, 'msg') and isinstance(record.msg, dict):
            # Clean up event logs
            if 'event' in record.msg:
                event_type = record.msg['event']
                if event_type == 'on_chat_model_stream':
                    # Only show token content for streaming
                    content = record.msg.get('data', {}).get('chunk', {}).get('content', '')
                    record.msg = f"Stream token: {content}"
                elif event_type == 'on_chain_end':
                    # Clean up chain end events
                    output = record.msg.get('data', {}).get('output', {})
                    if isinstance(output, dict) and 'messages' in output:
                        last_msg = output['messages'][-1] if output['messages'] else {}
                        record.msg = f"Chain output: {last_msg.get('content', '')}"
                    else:
                        record.msg = "Chain completed"
        return super().format(record)

class ColoredJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with colors for development."""
    
    COLORS = {
        'DEBUG': '\033[37m',  # White
        'INFO': '\033[32m',   # Green
        'WARNING': '\033[33m', # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[41m' # Red background
    }
    RESET = '\033[0m'

    def format(self, record):
        if isinstance(record.msg, dict):
            record.message = record.msg
        else:
            record.message = record.getMessage()
        
        # Add color in development
        if os.getenv('ENV', 'development') == 'development':
            color = self.COLORS.get(record.levelname, '')
            record.message = f"{color}{record.message}{self.RESET}"
        
        return super().format(record)

def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    is_dev = os.getenv('ENV', 'development') == 'development'
    
    # Define formatters based on environment
    if is_dev:
        log_format = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
        formatter_class = ColoredJsonFormatter
        stream_formatter = StreamFormatter(log_format)
    else:
        log_format = "%(timestamp)s %(level)s %(name)s %(message)s %(correlation_id)s"
        formatter_class = jsonlogger.JsonFormatter
        stream_formatter = StreamFormatter(log_format)

    log_config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "()": f"{formatter_class.__module__}.{formatter_class.__name__}",
                "format": log_format,
                "rename_fields": {
                    "levelname": "level",
                    "asctime": "timestamp"
                },
                "json_default": str,
                "timestamp": True
            },
            "stream": {
                "()": f"{StreamFormatter.__module__}.{StreamFormatter.__name__}",
                "format": "%(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "standard",
                "stream": sys.stdout,
            },
            "stream": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "stream",
                "stream": sys.stdout,
            }
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["console"],
                "level": "WARNING",  # Set root logger to WARNING to reduce noise
                "propagate": True,
            },
            "agents": {
                "handlers": ["console"],
                "level": level,
                "propagate": False,
            },
            "service": {
                "handlers": ["stream"],  # Use stream handler for service logs
                "level": level,
                "propagate": False,
            },
            # Explicitly configure uvicorn access logs
            "uvicorn.access": {
                "handlers": ["console"],
                "level": "WARNING",  # Reduce uvicorn access logs
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["console"],
                "level": "ERROR",  # Only show errors
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
    logger.info("Logging configured", extra={"correlation_id": correlation_id})

class CorrelationFilter(logging.Filter):
    """Add correlation ID to log records."""
    def __init__(self, correlation_id: str):
        super().__init__()
        self.correlation_id = correlation_id

    def filter(self, record):
        record.correlation_id = self.correlation_id
        return True