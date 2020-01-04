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

from models.model_builder import build_model
from dataset.dataset import Video_Dataset
from utils.misc import get_time_diff, save_scores
from utils.metric import Metric
from dataset.transform import *


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
    if cfg.TEST.SAVE_RESULTS:
        output = {}
        output["action_id"] = []
        for key in cfg.MODEL.NUM_CLASSES.keys():
            output[key] = []

    with torch.no_grad():
        for data, target, action_id in tqdm(data_loader):
            data = dict_to_device(data)

            if isinstance(target, dict):
                target = dict_to_device(target)

            out = model(data)

            if isinstance(target, dict):
                loss, batch_size = model.get_loss(criterion, target, out)
                metric.set_metrics(out, target, batch_size, loss)
            if cfg.TEST.SAVE_RESULTS:
                output["action_id"].extend([action_id])
                for key in out.keys():
                    output[key].extend([out[key]])

    test_loss, test_acc, conf_mat = metric.get_metrics()

    if cfg.TEST.SAVE_RESULTS:
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

    cfg.MODEL.USE_SOFTMAX = True

    logger.info("Initializing model...")
    model, criterion = build_model(cfg, modality, device)
    logger.info("Model initialized.")
    logger.info("----------------------------------------------------------")

    if cfg.TEST.PRE_TRAINED:
        pre_trained = cfg.TEST.PRE_TRAINED
    elif cfg.TRAIN.PRE_TRAINED:
        pre_trained = cfg.TRAIN.PRE_TRAINED
    else:
        logger.exception(
            "No pre-trained weights exist. Please set the PRE_TRAINED parameter for either TRAIN or TEST in config file."
        )

    logger.info("Loading pre-trained weights {}...".format(pre_trained))
    data_dict = torch.load(pre_trained, map_location="cpu")
    if cfg.NUM_GPUS > 1:
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
                    Rescale(cfg.DATA.TEST_SCALE_SIZE),
                    CenterCrop(cfg.DATA.TEST_CROP_SIZE),
                    Stack(m),
                    ToTensor(),
                    Normalize(cfg.DATA.RGB_MEAN, cfg.DATA.RGB_STD),
                ]
            )
        elif m == "Flow":
            test_transforms[m] = torchvision.transforms.Compose(
                [
                    Rescale(cfg.DATA.TEST_SCALE_SIZE),
                    CenterCrop(cfg.DATA.TEST_CROP_SIZE),
                    Stack(m),
                    ToTensor(),
                    Normalize(cfg.DATA.FLOW_MEAN, cfg.DATA.FLOW_STD),
                ]
            )
        elif m == "Audio":
            test_transforms[m] = torchvision.transforms.Compose(
                [Stack(m), ToTensor(is_audio=True)]
            )
    logger.info("No of files to test: {}".format(len(cfg.TEST.ANNOTATION_FILE)))
    logger.info("----------------------------------------------------------")

    assert len(cfg.TEST.ANNOTATION_FILE) == len(
        cfg.TEST.RESULTS_FILE
    ), "Number of annotations files to test ({}) and number of result files ({}) do not match".format(
        len(cfg.TEST.ANNOTATION_FILE), len(cfg.TEST.RESULTS_FILE)
    )

    for idx, annotation in enumerate(cfg.TEST.ANNOTATION_FILE):
        if cfg.TEST.VID_LIST:
            logger.info("Reading list of test videos...")
            with open(cfg.TEST.VID_LIST) as f:
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
            batch_size=cfg.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
        )
        logger.info("Done.")
        logger.info("----------------------------------------------------------")

        logger.info("{} action segments to be processed.".format(len(test_dataset)))
        logger.info("Inference in progress...")

        start_time = time.time()

        results = test(cfg, model, test_loader, criterion, modality, logger, device)

        logger.info("----------------------------------------------------------")
        logger.info("Test_Loss: {}".format(results[0]))
        logger.info("----------------------------------------------------------")
        logger.info("Accuracy Top {}:".format(cfg.VAL.TOPK))
        logger.info(json.dumps(results[1], indent=2))
        logger.info("----------------------------------------------------------")

        if cfg.TEST.SAVE_RESULTS:
            output_dict = results[3]
            if cfg.DATA.OUT_DIR:
                out_file = os.path.join(cfg.DATA.OUT_DIR, cfg.TEST.RESULTS_FILE[idx])
            else:
                out_file = os.path.join("./", cfg.TEST.RESULTS_FILE[idx])
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
