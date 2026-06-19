import os
import time
import logging
from typing import Optional


def setup_logger(name: str, log_dir: Optional[str] = None) -> logging.Logger:
    """Set up a unified logging configuration.

    Args:
        name: the name of the logger
        log_dir: the log file directory; if None, the default logs directory is used

    Returns:
        the configured logger instance
    """
    # Set the root logger level to DEBUG
    logging.getLogger().setLevel(logging.DEBUG)

    # Suppress verbose output from third-party libraries
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("accelerate").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("requests").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("langgraph").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    # Get or create the logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # The logger itself records DEBUG level and above
    logger.propagate = False  # Prevent log messages from propagating to the parent logger

    # If handlers already exist, do not add more
    if logger.handlers:
        return logger

    # Create the console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # The console only shows INFO level and above

    # Create the formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    # Create the file handler
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # The file records logs at DEBUG level and above
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# Predefined icons
SUCCESS_ICON = "✓"
ERROR_ICON = "✗"
WAIT_ICON = "🔄"
