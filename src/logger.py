import logging
import sys
from loguru import logger

class InterceptHandler(logging.Handler):
    """
    Intercept standard logging messages and route them to Loguru.
    Code from loguru documentation.
    """
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            if frame.f_back:
                frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

def setup_logger():
    """Configure Loguru to intercept uvicorn and fastapi logs, and format them clearly."""
    logging.root.handlers = [InterceptHandler()]
    logging.root.setLevel(logging.INFO)

    # Remove all loguru's default handlers
    logger.remove()
    
    # Configure our own handler to sys.stderr with nice formatting
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True,
    )

    # Remove every other logger's handlers and propagate to root logger
    for name in logging.root.manager.loggerDict.keys():
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True
    
    # Let uvicorn loggers intercept gracefully
    for name in ["uvicorn.access", "uvicorn.error", "uvicorn"]:
        logging_logger = logging.getLogger(name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False

    return logger

log = setup_logger()
