#!/usr/bin/env python

import argparse
import numpy as np
import os
import time
from tensorboardX import SummaryWriter
from omegaconf import OmegaConf

from models.model import Model


def parse_args():
    parser = argparse.ArgumentParser(description="main.py")
    parser.add_argument(
        "--cfg",
        dest="cfg",
        help="cfg model file (/path/to/model_config.yaml)",
        default="./config/config.yaml",
        type=str,
    )
    # parser.add_argument(
    #     "--checkpoint",
    #     dest="weights",
    #     help="weights model file (/path/to/model_weights.pkl)",
    #     default=config.weights,
    #     type=str,
    # )
    # parser.add_argument(
    #     "--dataset_root",
    #     dest="dataset_root",
    #     help="path_to_dataset_root",
    #     type=str,
    # )
    # parser.add_argument(
    #     "--out_path",
    #     dest="out_path",
    #     help="path_to_save_detections",
    #     default=config.out_path,
    #     type=str,
    # )
    
    return parser.parse_args()

def main(args):

    cfg = OmegaConf.load(args.cfg)
    
    print(cfg.pretty())

if __name__ == "__main__":
    args = parse_args()
    main(args)