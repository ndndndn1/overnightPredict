"""Structured logging configuration."""

import logging
import sys
from typing import Any, Optional

import structlog
from structlog.typing import EventDict, WrappedLogger


def add_timestamp(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add timestamp to log events."""
    from datetime import datetime

    event_dict["timestamp"] = datetime.utcnow().isoformat()
    return event_dict


def add_caller_info(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add caller information to log events."""
    import inspect

    frame = inspect.currentframe()
    if frame:
        # Walk up the stack to find the actual caller
        for _ in range(10):
            frame = frame.f_back
            if frame is None:
                break
            module = frame.f_globals.get("__name__", "")
            if not module.startswith("structlog") and not module.startswith("logging"):
                event_dict["caller"] = f"{module}:{frame.f_lineno}"
                break

    return event_dict


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """Set up structured logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: If True, output logs in JSON format.
        log_file: Optional file path for log output.
    """
    # Configure processors
    processors: list[Any] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        add_timestamp,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        logging.getLogger().addHandler(file_handler)


def get_logger(name: Optional[str] = None, **initial_context: Any) -> structlog.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (usually module name).
        **initial_context: Initial context to bind to the logger.

    Returns:
        A bound structlog logger.
    """
    logger = structlog.get_logger(name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger


class LoggerAdapter:
    """Adapter to use structlog with async context managers."""

    def __init__(self, logger: structlog.BoundLogger):
        self._logger = logger
        self._context: dict[str, Any] = {}

    def bind(self, **kwargs: Any) -> "LoggerAdapter":
        """Bind additional context."""
        self._context.update(kwargs)
        self._logger = self._logger.bind(**kwargs)
        return self

    def unbind(self, *keys: str) -> "LoggerAdapter":
        """Remove context keys."""
        for key in keys:
            self._context.pop(key, None)
        self._logger = self._logger.unbind(*keys)
        return self

    def info(self, event: str, **kwargs: Any) -> None:
        """Log info message."""
        self._logger.info(event, **kwargs)

    def debug(self, event: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._logger.debug(event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._logger.warning(event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        """Log error message."""
        self._logger.error(event, **kwargs)

    def exception(self, event: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        self._logger.exception(event, **kwargs)

    async def __aenter__(self) -> "LoggerAdapter":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        if exc_val:
            self.exception("Context manager exited with exception", exc_type=str(exc_type))
