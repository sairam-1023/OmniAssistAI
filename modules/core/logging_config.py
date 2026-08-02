"""
Structured logging setup for OmniAssist AI.

Call configure_logging() once at application startup (we'll do this in
app/main.py in Week 7). Every module then gets its own named logger via
get_logger(__name__), so log lines are traceable to their source.
"""

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """
    Set up the root logging configuration. Call this exactly once,
    at application startup.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger for a specific module.

    Usage, at the top of any module file:
        from modules.core.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("Something happened")
    """
    return logging.getLogger(name)