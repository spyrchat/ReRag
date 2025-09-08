import logging
from pathlib import Path

# Ensure log directory exists
Path("logs").mkdir(exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger instance writing to logs/agent.log.
    Ensures handlers are not duplicated.
    Args:
        name (str): Logger name, usually __name__.
    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Only add handler if none exist for this logger
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith("agent.log")
               for h in logger.handlers):
        handler = logging.FileHandler("logs/agent.log")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

# Example usage in other modules:
# logger = get_logger(__name__)
# logger.info("Logger is working!")
