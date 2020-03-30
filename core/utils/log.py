import os
import logging
from tensorboardX import SummaryWriter


def setup_log(modality):
    """
    Helper function to setup the log and initialize SummaryWriter

    Args
    ----------
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

    log_dir = os.getcwd()
    logger = logging.getLogger(__name__)

    writer = SummaryWriter(logdir=log_dir)

    return logger, writer
