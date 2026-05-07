import logging
import os


def get_future_aware_logger():
    """
    Returns a module-specific logger that writes to future_aware.log
    in the same directory as this file.

    Log level: DEBUG (captures info, warnings, errors, and debug messages).
    """
    logger = logging.getLogger("future_aware")
    if logger.handlers:
        # Logger already configured
        return logger

    # Capture everything from DEBUG and above
    logger.setLevel(logging.CRITICAL + 1) #logger.setLevel(logging.DEBUG) # logger.setLevel(logging.DEBUG) #  logger.setLevel(logging.CRITICAL + 1) #

    # Log file path: future_aware/future_aware.log
    log_dir = os.path.dirname(__file__)
    log_path = os.path.join(log_dir, "future_aware.log")

    # File handler for this module
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)

    # Formatter: timestamp, level, file, line, message
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    # Do not propagate to root logger (avoid duplicate logs on stdout)
    logger.propagate = False

    return logger



