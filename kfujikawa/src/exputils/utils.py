import time
from contextlib import contextmanager

from loguru import logger


@contextmanager
def timer(message: str, level: str = "DEBUG"):
    """Function to measure elapsed time.

    Parameters
    ----------
    message : str
        Message
    level : str, optional
        Logging level, by default "DEBUG"
    """

    t0 = time.time()
    logger.log(level, f"[START] {message}")
    yield
    logger.log(level, f"[DONE] {message} ({time.time() - t0:.1f}s)")
