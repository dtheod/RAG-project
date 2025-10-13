from loguru import logger
import sys

def level_filter(level):
    def is_level(record):
        return record["level"].name == level
    return is_level

logger.add(sys.stderr, format="{time} | {level} | {message}")
logger.add("./outputs/loguru.log")