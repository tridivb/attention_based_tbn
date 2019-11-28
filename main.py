#!/usr/bin/env python

import argparse
import numpy as np
import os
import time
import torch
from tensorboardX import SummaryWriter
from omegaconf import OmegaConf
from datetime import datetime

from train import train
from test import test
from utils.log import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="main.py")
    parser.add_argument(
        "--cfg",
        dest="cfg",
        help="cfg model file (/path/to/model_config.yaml)",
        default="./config/config.yaml",
        type=str,
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
    if cfg.LOG_DIR == "":
        log_root = "./log"
    else:
        log_root = cfg.LOG_DIR

    # log_dir = os.path.join(log_root, log_dir)
    # os.makedirs(log_dir, exist_ok=True)
    # log_file = os.path.join(log_dir, "tbn.log")

    # logger = setup_logger(log_file)
    # logger.info("Initializing the pipeline...")
    # logger.info(cfg.pretty())
    # print("----------------------------------------------------------")
    logger = None

    torch.hub.set_dir("./weights")

    if cfg.TRAIN.TRAIN_ENABLE:
        train(cfg, logger, modality)


if __name__ == "__main__":
    args = parse_args()
    main(args)
