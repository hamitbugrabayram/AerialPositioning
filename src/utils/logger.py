"""Structured logging configuration for the aerial positioning system.

This module provides a unified logger factory that creates standardized,
structured loggers for different components of the system.
"""

import logging
import sys


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
    reset = "\x1b[0m"
    
    format_str = "[%(levelname)s] %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: blue + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset,
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


def get_logger(name: str) -> logging.Logger:
    """Creates or retrieves a standardized logger instance.

    Args:
        name: The name of the logger, typically `__name__`.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(CustomFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger
