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
    # parser.add_argument(
    #     "mode",
    #     choices={"train", "test"},
    #     help="mode to process",
    #     type=str,
    # )
    parser.add_argument(
        "--cfg",
        dest="cfg",
        help="cfg model file (/path/to/model_config.yaml)",
        default="./config/config.yaml",
        type=str,
    )
    # parser.add_argument(
    #     "--wts",
    #     dest="weights",
    #     help="weights model file (/path/to/model_weights.pkl)",
    #     default=config.weights,
    #     type=str,
    # )
    # parser.add_argument(
    #     "--top_predictions",
    #     dest="top_predictions",
    #     help="Number of predictions to store",
    #     default=100,
    #     type=int,
    # )
    # parser.add_argument(
    #     "--video_root",
    #     dest="video_root",
    #     help="path_to_video_root",
    #     default=config.video_root,
    #     type=str,
    # )
    # parser.add_argument(
    #     "--video_list",
    #     dest="video_list",
    #     help="path_to_list_of_videos",
    #     default=config.video_list,
    #     type=str,
    # )
    # parser.add_argument(
    #     "--out_path",
    #     dest="out_path",
    #     help="path_to_save_detections",
    #     default=config.out_path,
    #     type=str,
    # )
    # parser.add_argument(
    #     "--model",
    #     choices={"vgg11", "vgg11bn", "vgg16", "vgg16bn", "resnet18", "resnet34", "resnet50", "resnet101", "bninception"},
    #     help="model name",
    #     default="vgg16bn",
    #     type=str,
    # )
    # parser.add_argument(
    #     "--target",
    #     choices={"verb", "noun", "action"},
    #     help="target to classify",
    #     default="action",
    #     type=str,
    # )
    

    return parser.parse_args()

def main(args):

    cfg = OmegaConf.load(args.cfg)
    
    # if args.target == "verb":
    #     num_classes = 125
    # elif args.target == "noun":
    #     num_classes = 352
    # elif args.target == "action":
    #     num_classes = [125, 352]

    # model = Model(args.model, num_classes, args.mode)
    # out = model((torch.zeros(1, 3, 224, 224), torch.zeros(1, 16, 224, 224), torch.zeros(1, 1, 224, 224)))
    print(cfg.pretty())

if __name__ == "__main__":
    args = parse_args()
    main(args)