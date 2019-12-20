import os
import time
import logging
from datetime import datetime
from tensorboardX import SummaryWriter


def setup_handler(logger, handler, fmt, datefmt):
    """
    Helper function to setup console and file handlers for logger

    Args
    ----------
    logger: logging.logger
        Python logger
    handler: logging.handler
        Type of logging handler
    fmt: str
        Format of logging each line
    datefmt: str
        Date time format for each line of logging

    """

    handler.setLevel(logging.INFO)
    h_format = logging.Formatter(fmt=fmt, datefmt=datefmt)
    handler.setFormatter(h_format)

    logger.addHandler(handler)


def setup_logger(log_file):
    """
    Helper function to setup console and file handlers for logger

    Args
    ----------
    log_file: str
        Absolute path of log file 
    
    Returns
    ----------
    logger: logging.logger
        The logger to use

    """

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


def setup_log(cfg, modality):
    """
    Helper function to setup the log and initialize SummaryWriter

    Args
    ----------
    cfg: dict
        Dictionary of config parameters
    modality: list
        List of input modalities
    
    Returns
    ----------
    logger: logging.logger
        The logger to use
    writer: SummaryWriter
        Tensorboard summary writer to plot metrics

    """
    # Create log directory
    timestamp = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = "run_{}_{}_{}_{}".format(
        cfg.MODEL.ARCH, cfg.DATA.DATASET, "-".join(modality), timestamp
    )

    log_dir = os.path.join(cfg.DATA.OUT_DIR, "log", log_dir)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "tbn.log")

    logger = setup_logger(log_file)

    writer = SummaryWriter(logdir=log_dir)

    return logger, writer
