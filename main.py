#!/usr/bin/env python

import argparse
import torch
from omegaconf import OmegaConf
import numpy as np
import hydra

from tools.train import run_trainer
from tools.test import run_tester
from utils.log import setup_log
from utils.misc import get_modality

torch.multiprocessing.set_sharing_strategy("file_system")


@hydra.main(config_path="config/config.yaml")
def main(cfg):

    np.random.seed(cfg.data.manual_seed)
    torch.manual_seed(cfg.data.manual_seed)

    modality = get_modality(cfg)

    logger, writer = setup_log(modality)

    logger.info("Initializing the pipeline...")
    logger.info(cfg.pretty())
    logger.info("Modality: {}".format(modality))
    logger.info("----------------------------------------------------------")

    try:
        if cfg.train.train_enable:
            logger.info("Training the model.")
            run_trainer(cfg, logger, modality, writer)

        if cfg.test.test_enable:
            logger.info("Evaluating the model.")
            run_tester(cfg, logger, modality)
    except Exception as e:
        logger.exception(e)


if __name__ == "__main__":

    main()
