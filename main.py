#!/usr/bin/env python

import argparse
import numpy as np
import os
import time
import torch
from omegaconf import OmegaConf
from datetime import datetime

from tools.train import run_trainer
from tools.test import run_tester
from utils.log import setup_log
from utils.misc import get_modality


def parse_args():
    parser = argparse.ArgumentParser(description="main.py")
    parser.add_argument(
        "cfg", help="cfg model file (/path/to/config.yaml)", type=str,
    )

    return parser.parse_args()


def main(args):

    cfg = OmegaConf.load(args.cfg)

    modality = get_modality(cfg)

    logger, writer = setup_log(cfg, modality)

    logger.info("Initializing the pipeline...")
    logger.info(cfg.pretty())
    logger.info("----------------------------------------------------------")

    try:
        if cfg.TRAIN.TRAIN_ENABLE:
            logger.info("Training the model.")
            run_trainer(cfg, logger, modality, writer)

        if cfg.TEST.TEST_ENABLE:
            logger.info("Evaluating the model.")
            run_tester(cfg, logger, modality)
    except Exception as e:
        logger.exception(e)


if __name__ == "__main__":
    args = parse_args()
    main(args)
