#!/usr/bin/env python

import os
import numpy as np
import torch
import torchvision
import pandas as pd
from torch.utils.data.dataloader import default_collate
from omegaconf import OmegaConf

from core.models import build_model
from core.dataset import Video_Dataset
from core.utils import get_modality
from core.dataset.transform import *


def infer(
    cfg, model, data, target, device=torch.device("cuda")
):
    """
    Evaluate the model

    Args
    ----------
    cfg: dict
        Dictionary of config parameters
    model: torch.nn.model
        Model to train
    data_loader: DataLoader
        Data loader to iterate over the data
    criterion: loss
        Loss function to use
    modality: list
        List of input modalities
    logger: logger
        Python logger
    device: torch.device, default = torch.device("cuda")
        Torch device to use

    Returns
    ----------
    test_loss: dict
        Dictionary of losses for each class and sum of all losses
    test_acc: dict
        Accuracy of each type of class
    confusion_matrix: Tensor
        Array of the confusion matrix over the test set
    output: dict
        Dictionary of model output over the test set

    """

    dict_to_device = TransferTensorDict(device)

    model.eval()

    with torch.no_grad():
        data, target = dict_to_device(data), dict_to_device(target)
        out = model(data)

    return out


def initialize(config_file):
    """
    Initialize model , data loaders, loss function, optimizer and evaluate the model

    Args
    ----------
    cfg: dict
        Dictionary of config parameters
    modality: list
        List of input modalities

    """

    cfg = OmegaConf.load(config_file)

    np.random.seed(cfg.data.manual_seed)
    torch.manual_seed(cfg.data.manual_seed)

    modality = get_modality(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Initializing model...")
    model, _, num_gpus = build_model(cfg, modality, device)
    print("Model initialized.")
    print("----------------------------------------------------------")

    if cfg.test.pre_trained:
        if os.path.exists(cfg.test.pre_trained):
            print("Loading pre-trained weights {}...".format(cfg.test.pre_trained))
            data_dict = torch.load(cfg.test.pre_trained, map_location="cpu")
            if num_gpus > 1:
                model.module.load_state_dict(data_dict["model"])
            else:
                model.load_state_dict(data_dict["model"])
            print("Done.")
            print("----------------------------------------------------------")
        else:
            raise Exception(f"{cfg.test.pre_trained} file not found.")

    transforms = {}
    for m in modality:
        if m == "RGB":
            transforms[m] = torchvision.transforms.Compose(
                [
                    Rescale(cfg.data.test_scale_size),
                    CenterCrop(cfg.data.test_crop_size),
                    Stack(m),
                    ToTensor(),
                    Normalize(cfg.data.rgb.mean, cfg.data.rgb.std),
                ]
            )
        elif m == "Flow":
            transforms[m] = torchvision.transforms.Compose(
                [
                    Rescale(cfg.data.test_scale_size),
                    CenterCrop(cfg.data.test_crop_size),
                    Stack(m),
                    ToTensor(),
                    Normalize(cfg.data.flow.mean, cfg.data.flow.std),
                ]
            )
        elif m == "Audio":
            transforms[m] = torchvision.transforms.Compose(
                [Stack(m), ToTensor(is_audio=True)]
            )


    if cfg.test.vid_list:
        print("Reading list of test videos...")
        with open(os.path.join("./", cfg.test.vid_list)) as f:
            test_list = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        print("Done.")
        print("----------------------------------------------------------")
    else:
        test_list = None

    print("Creating the dataset using {}...".format(cfg.test.annotation_file[0]))
    dataset = Video_Dataset(
        cfg,
        test_list,
        cfg.test.annotation_file[0],
        modality,
        transform=transforms,
        mode="test",
    )
    print("Done.")
    print("----------------------------------------------------------")

    return cfg, model, dataset, device