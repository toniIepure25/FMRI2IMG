from logging import getLogger, basicConfig, INFO, Formatter, StreamHandler
import sys

def get_logger(name: str):
    logger = getLogger(name)
    if not logger.handlers:
        handler = StreamHandler(sys.stdout)
        handler.setFormatter(Formatter("[%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(INFO)
    return logger
