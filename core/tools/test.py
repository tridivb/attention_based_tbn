#!/usr/bin/env python

import os
import time
import json
import numpy as np
import torch
import torchvision
import pandas as pd
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader

from core.models import build_model
from core.dataset import Video_Dataset
from core.utils import get_time_diff, save_scores, Metric
from core.dataset.transform import *


def test(
    cfg, model, data_loader, criterion, modality, logger, device=torch.device("cuda")
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

    no_batches = round(len(data_loader.dataset) / data_loader.batch_size)
    dict_to_device = TransferTensorDict(device)
    metric = Metric(cfg, no_batches, device)

    model.eval()
    if cfg.test.save_results:
        output = {}
        output["action_id"] = []
        for key in cfg.model.num_classes.keys():
            output[key] = []

    with torch.no_grad():
        for data, target, action_id in tqdm(data_loader):
            data, target = dict_to_device(data), dict_to_device(target)

            out = model(data)

            if isinstance(target["class"], dict):
                loss, batch_size = model.get_loss(criterion, target, out)
                metric.set_metrics(out, target, batch_size, loss)

            if cfg.test.save_results:
                output["action_id"].extend([action_id])
                for key in cfg.model.num_classes.keys():
                    output[key].extend([out[key]])

    test_loss, test_acc, conf_mat = metric.get_metrics()

    if cfg.test.save_results:
        return (test_loss, test_acc, conf_mat, output)
    else:
        return (test_loss, test_acc, conf_mat)


def run_tester(cfg, logger, modality):
    """
    Initialize model , data loaders, loss function, optimizer and evaluate the model

    Args
    ----------
    cfg: dict
        Dictionary of config parameters
    logger: logger
        Python logger
    modality: list
        List of input modalities

    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Initializing model...")
    model, criterion, num_gpus = build_model(cfg, modality, device)
    logger.info("Model initialized.")
    logger.info(model)
    logger.info("----------------------------------------------------------")

    if cfg.test.pre_trained:
        pre_trained = cfg.test.pre_trained
    else:
        logger.exception(
            "No pre-trained weights exist. Please set the pre_trained parameter for test in config file."
        )

    logger.info("Loading pre-trained weights {}...".format(pre_trained))
    data_dict = torch.load(pre_trained, map_location="cpu")
    if num_gpus > 1:
        model.module.load_state_dict(data_dict["model"])
    else:
        model.load_state_dict(data_dict["model"])
    logger.info("Done.")
    logger.info("----------------------------------------------------------")

    test_transforms = {}
    for m in modality:
        if m == "RGB":
            test_transforms[m] = torchvision.transforms.Compose(
                [
                    Rescale(cfg.data.test_scale_size),
                    CenterCrop(cfg.data.test_crop_size),
                    Stack(m),
                    ToTensor(),
                    Normalize(cfg.data.rgb.mean, cfg.data.rgb.std),
                ]
            )
        elif m == "Flow":
            test_transforms[m] = torchvision.transforms.Compose(
                [
                    Rescale(cfg.data.test_scale_size),
                    CenterCrop(cfg.data.test_crop_size),
                    Stack(m),
                    ToTensor(),
                    Normalize(cfg.data.flow.mean, cfg.data.flow.std),
                ]
            )
        elif m == "Audio":
            test_transforms[m] = torchvision.transforms.Compose(
                [Stack(m), ToTensor(is_audio=True)]
            )
    logger.info("No of files to test: {}".format(len(cfg.test.annotation_file)))
    logger.info("----------------------------------------------------------")

    if cfg.test.save_results:
        assert len(cfg.test.annotation_file) == len(
            cfg.test.results_file
        ), "Number of annotations files to test ({}) and number of result files ({}) do not match".format(
            len(cfg.test.annotation_file), len(cfg.test.results_file)
        )

    start_time = time.time()

    for idx, annotation in enumerate(cfg.test.annotation_file):
        if cfg.test.vid_list:
            logger.info("Reading list of test videos...")
            file_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            with open(os.path.join(file_dir, cfg.test.vid_list)) as f:
                test_list = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
            logger.info("Done.")
            logger.info("----------------------------------------------------------")
        else:
            test_list = None

        logger.info("Creating the dataset using {}...".format(annotation))
        test_dataset = Video_Dataset(
            cfg,
            test_list,
            annotation,
            modality,
            transform=test_transforms,
            mode="test",
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.test.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )
        logger.info("Done.")
        logger.info("----------------------------------------------------------")

        logger.info("{} action segments to be processed.".format(len(test_dataset)))
        logger.info("Inference in progress...")

        results = test(cfg, model, test_loader, criterion, modality, logger, device)

        logger.info("----------------------------------------------------------")
        logger.info("Test_Loss: {}".format(results[0]))
        logger.info("----------------------------------------------------------")
        logger.info("Accuracy Top {}:".format(cfg.val.topk))
        logger.info(json.dumps(results[1], indent=2))
        logger.info("----------------------------------------------------------")

        if cfg.test.save_results:
            output_dict = results[3]
            if cfg.out_dir:
                out_file = os.path.join(
                    cfg.out_dir, "inferences", cfg.test.results_file[idx]
                )
            else:
                out_file = os.path.join("./inferences", cfg.test.results_file[idx])
            try:
                save_scores(output_dict, out_file)
                logger.info("Saved results to {}".format(out_file))
            except Exception as e:
                logger.exception(e)

    hours, minutes, seconds = get_time_diff(start_time, time.time())
    logger.info(
        "Inference time: {} hours, {} minutes, {} seconds,".format(
            hours, minutes, seconds
        )
    )
