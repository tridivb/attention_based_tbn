import logging


def setup_handler(logger, handler, fmt, datefmt):
    # Setup console log format
    handler.setLevel(logging.INFO)
    h_format = logging.Formatter(fmt=fmt, datefmt=datefmt)
    handler.setFormatter(h_format)

    logger.addHandler(handler)


def setup_logger(log_file):

    logger = logging.getLogger(__name__)

    logger.setLevel(logging.DEBUG)

    setup_handler(
        logger,
        logging.StreamHandler(),
        fmt="%(levelname)s : %(asctime)s : %(message)s",
        datefmt="%H:%M:%S",
    )
    setup_handler(
        logger,
        logging.FileHandler(log_file),
        fmt="%(levelname)s : %(asctime)s : %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )

    return logger
