"""Structured logging configuration for the aerial positioning system.

This module provides a unified logger factory that creates standardized,
structured loggers for different components of the system.
"""

import logging
import sys
from typing import Any, cast

_FALLBACK_LEVEL = 60
logging.addLevelName(_FALLBACK_LEVEL, "FALLBACK")

class CustomFormatter(logging.Formatter):
    """Custom formatter providing structured and aligned log output.

    Attributes:
        FORMATS (dict): Mapping of logging levels to their string formats.
    """

    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    magenta = "\x1b[35;20m"
    reset = "\x1b[0m"
    
    format_str = "[%(levelname)s] %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: blue + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset,
        _FALLBACK_LEVEL: magenta + format_str + reset,
    }

    def format(self, record: logging.LogRecord) -> str:
        """Formats a single log record.

        Args:
            record: The log record to format.

        Returns:
            The formatted log string.
        """
        log_fmt = self.FORMATS.get(record.levelno, self.format_str)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class AppLogger(logging.Logger):
    """Extended logger class supporting custom fallback level.

    Attributes:
        name: The name of the logger.
    """

    def fallback(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Logs a message with level FALLBACK.

        Args:
            msg: The message to log.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        if self.isEnabledFor(_FALLBACK_LEVEL):
            self._log(_FALLBACK_LEVEL, msg, args, **kwargs)

def get_logger(name: str) -> AppLogger:
    """Creates or retrieves a standardized logger instance.

    Args:
        name: The name of the logger, typically __name__.

    Returns:
        A configured AppLogger instance.
    """
    logging.setLoggerClass(AppLogger)
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(CustomFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return cast(AppLogger, logger)
