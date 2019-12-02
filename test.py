#!/usr/bin/env python

import os
import time
import json
import numpy as np
import torch
import torchvision
from torch.utils.data.dataloader import DataLoader
from tensorboardX import SummaryWriter

from models.model_builder import build_model
from utils.dataset import Video_Dataset
from utils.misc import get_time_diff
from utils.metric import Metric
from utils.transform import *


def test(
    cfg, model, data_loader, criterion, modality, logger, device=torch.device("cuda")
):
    no_batches = round(len(data_loader.dataset) / data_loader.batch_size)
    dict_to_device = TransferTensorDict(device)
    metric = Metric()

    model.eval()
    test_loss = 0
    test_acc = {}
    precision = {}
    recall = {}
    confusion_matrix = {}
    predictions = []
    for cls, no_cls in cfg.MODEL.NUM_CLASSES.items():
        test_acc[cls] = [0] * (len(cfg.VAL.TOPK))
        confusion_matrix[cls] = np.zeros((no_cls, no_cls))
        precision[cls] = 0
        recall[cls] = 0

    with torch.no_grad():
        for data, target in data_loader:
            data = dict_to_device(data)

            if isinstance(target, torch.tensor):
                target = target.to(device)
            elif isinstance(target, dict):
                target = dict_to_device(target)

            out = model(data)

            if target != -1:
                loss = model.get_loss(criterion, target, out)
                test_loss += loss.item()
                for cls in test_acc.keys():
                    acc, conf_mat, prec, rec = metric.calculate_metrics(
                        out[cls], target[cls], topk=cfg.VAL.TOPK
                    )
                    test_acc[cls] = [x + y for x, y in zip(test_acc[cls], acc)]
                    precision[cls] += prec
                    recall[cls] += rec
                    confusion_matrix[cls] += conf_mat
            else:
                for cls in target.keys():
                    _, preds = out[cls].max(1)
                    predictions.extend(preds.tolist())

    if test_loss > 0:
        test_loss /= no_batches
        for cls in test_acc.keys():
            test_acc[cls] = [round(x / no_batches, 2) for x in test_acc[cls]]
            precision[cls] /= no_batches
            recall[cls] /= no_batches
        return test_loss, test_acc, confusion_matrix, precision, recall
    else:
        return predictions


def save_predictions(results):
    raise Exception("Not implemented")


def run_tester(cfg, logger, modality):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    modality = []

    model = build_model(cfg, modality)

    if cfg.MODEL.CHECKPOINT:
        model.load_state_dict(torch.load(cfg.MODEL.CHECKPOINT))

    criterion = torch.nn.CrossEntropyLoss()

    model, criterion = model.to(device), criterion.to(device)

    with open(cfg.TEST.VID_LIST) as f:
        test_list = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

    test_transforms = {}
    for m in modality:
        if m == "RGB":
            test_transforms[m] = torchvision.transforms.Compose(
                [
                    Rescale(cfg.DATA.TEST_SCALE_SIZE, is_flow=False),
                    CenterCrop(cfg.DATA.TEST_CROP_SIZE),
                    Stack(m),
                    ToTensor(),
                    Normalize(cfg.DATA.RGB_MEAN, cfg.DATA.RGB_STD),
                ]
            )
        elif m == "Flow":
            test_transforms[m] = torchvision.transforms.Compose(
                [
                    Rescale(cfg.DATA.TEST_SCALE_SIZE, is_flow=True),
                    CenterCrop(cfg.DATA.TEST_CROP_SIZE),
                    Stack(m),
                    ToTensor(),
                    Normalize(cfg.DATA.FLOW_MEAN, cfg.DATA.FLOW_STD),
                ]
            )
        elif m == "Audio":
            test_transforms[m] = torchvision.transforms.Compose([Stack(m), ToTensor()])

    test_dataset = Video_Dataset(
        cfg, test_list, modality, transform=test_transforms, mode="train"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )

    start_time = time.time()

    results = test(cfg, model, test_loader, criterion, modality, logger, device)

    if isinstance(results, tuple):
        print("----------------------------------------------------------")
        print("Test_Loss: {:5f}".format(results[0]))
        print("----------------------------------------------------------")
        print("Accuracy Top {}:".format(cfg.VAL.TOPK))
        print(json.dumps(results[1], indent=2))
        print("Precision: {:.2f}".format(json.dumps(results[2], indent=2)))
        print("Recall: {:.2f}".format(json.dumps(results[3], indent=2)))
        print("----------------------------------------------------------")
    else:
        save_predictions(results)

    hours, minutes, seconds = get_time_diff(start_time, time.time())
    print(
        "Inference time: {} hours, {} minutes, {} seconds,".format(
            hours, minutes, seconds
        )
    )
