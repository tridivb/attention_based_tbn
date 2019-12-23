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
    test_loss,: float
        Overall validation loss
    test_acc: dict
        Accuracy of each type of class
    confusion_matrix: np.ndarray
        Array of the confusion matrix over the validation set
    results: dict
        Dictionary of model output over the test set

    """

    no_batches = round(len(data_loader.dataset) / data_loader.batch_size)
    dict_to_device = TransferTensorDict(device)
    metric = Metric()

    model.eval()
    test_loss = {"total": 0}
    test_acc = {}
    confusion_matrix = {}
    results = {}
    results["action_id"] = []
    for cls, no_cls in cfg.MODEL.NUM_CLASSES.items():
        test_acc[cls] = [0] * (len(cfg.VAL.TOPK))
        confusion_matrix[cls] = torch.zeros((no_cls, no_cls), device=device)
        results[cls] = []
        test_loss[cls] = 0

    with torch.no_grad():
        for data, target, action_id in tqdm(data_loader):
            data = dict_to_device(data)

            if isinstance(target, dict):
                target = dict_to_device(target)
            else:
                target = target.to(device)

            out = model(data)

            if isinstance(target, dict):
                loss = model.get_loss(criterion, target, out)
                test_loss["total"] += loss["total"].item()
                for cls in test_acc.keys():
                    acc, conf_mat = metric.calculate_metrics(
                        out[cls], target[cls], device, topk=cfg.VAL.TOPK
                    )
                    test_acc[cls] = [x + y for x, y in zip(test_acc[cls], acc)]
                    confusion_matrix[cls] += conf_mat
                    test_loss[cls] += loss[cls].item()

            results["action_id"].extend([action_id.numpy()])
            for cls in out.keys():
                results[cls].extend(
                    [out[cls].cpu().numpy() if out[cls].is_cuda else out[cls].numpy()]
                )

    if test_loss["total"] > 0:
        test_loss["total"] = round(test_loss["total"] / no_batches, 5)
        for cls in test_acc.keys():
            test_acc[cls] = [round(x / no_batches, 2) for x in test_acc[cls]]
            test_loss[cls] = round(test_loss[cls] / no_batches, 5)
            if device.type == "cuda":
                confusion_matrix[cls] = confusion_matrix[cls].cpu()
            confusion_matrix[cls] = confusion_matrix[cls].numpy()
        return test_loss, test_acc, confusion_matrix, results
    else:
        return results


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

    logger.info("Reading list of test videos...")
    if cfg.TEST.VID_LIST:
        with open(cfg.TEST.VID_LIST) as f:
            test_list = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
    else:
        if cfg.TEST.ANNOTATION_FILE.endswith("csv"):
            df = pd.read_csv(cfg.TEST.ANNOTATION_FILE)
        elif cfg.TEST.ANNOTATION_FILE.endswith("pkl"):
            df = pd.read_pickle(cfg.TEST.ANNOTATION_FILE)
        test_list = df["video_id"].unique()

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

    logger.info("Creating the dataset...")
    test_dataset = Video_Dataset(
        cfg,
        test_list,
        cfg.TEST.ANNOTATION_FILE,
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

    if isinstance(results, tuple):
        logger.info("----------------------------------------------------------")
        logger.info("Test_Loss: {}".format(results[0]))
        logger.info("----------------------------------------------------------")
        logger.info("Accuracy Top {}:".format(cfg.VAL.TOPK))
        logger.info(json.dumps(results[1], indent=2))
        logger.info("----------------------------------------------------------")
        results_dict = results[3]
    else:
        results_dict = results

    if cfg.TEST.SAVE_RESULTS:
        if cfg.DATA.OUT_DIR:
            out_file = os.path.join(cfg.DATA.OUT_DIR, cfg.TEST.RESULTS_FILE)
        else:
            out_file = os.path.join("./", cfg.TEST.RESULTS_FILE)
        try:
            save_scores(results_dict, out_file)
            logger.info("Saved results to {}".format(out_file))
        except Exception as e:
            logger.exception(e)

    hours, minutes, seconds = get_time_diff(start_time, time.time())
    logger.info(
        "Inference time: {} hours, {} minutes, {} seconds,".format(
            hours, minutes, seconds
        )
    )
