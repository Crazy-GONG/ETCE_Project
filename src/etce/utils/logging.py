import logging


def make_logger(path):
    logger = logging.getLogger('etce.phase1')
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    handler = logging.FileHandler(path, mode='w', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    return logger
