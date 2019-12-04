#!/usr/bin/env python

import argparse
import numpy as np
import os
import time
import torch
from tensorboardX import SummaryWriter
from omegaconf import OmegaConf
from datetime import datetime

from train import run_trainer
from test import run_tester
from utils.log import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="main.py")
    parser.add_argument(
        "cfg", help="cfg model file (/path/to/config.yaml)", type=str,
    )

    return parser.parse_args()


def main(args):

    cfg = OmegaConf.load(args.cfg)

    modality = []

    if cfg.DATA.USE_RGB:
        modality.append("RGB")
    if cfg.DATA.USE_FLOW:
        modality.append("Flow")
    if cfg.DATA.USE_AUDIO:
        modality.append("Audio")

    # Create log directory
    timestamp = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = "run_{}_{}_{}_{}".format(
        cfg.MODEL.ARCH, cfg.DATA.DATASET, "-".join(modality), timestamp
    )

    log_root = cfg.LOG_DIR if cfg.LOG_DIR else "./log"

    log_dir = os.path.join(log_root, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "tbn.log")
    writer = SummaryWriter(logdir=log_dir)

    os.makedirs(cfg.DATA.OUT_DIR, exist_ok=True)

    logger = setup_logger(log_file)
    logger.info("Initializing the pipeline...")
    logger.info(cfg.pretty())
    logger.info("----------------------------------------------------------")

    try:
        if cfg.TRAIN.TRAIN_ENABLE:
            logger.info("Training the model.")
            run_trainer(cfg, logger, modality, writer)
    except Exception as e:
        logger.exception(e)

    try:
        if cfg.TEST.TEST_ENABLE:
            logger.info("Evaluating the model.")
            run_tester(cfg, logger, modality)
    except Exception as e:
        logger.exception(e)


if __name__ == "__main__":
    args = parse_args()
    main(args)
