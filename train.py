#!/usr/bin/env python

import os
import time
import json
import numpy as np
import torch
import torch.optim as optim
import torchvision
from torch.utils.data.dataloader import DataLoader
from tensorboardX import SummaryWriter

from models.model_builder import build_model
from utils.dataset import Video_Dataset
from utils.misc import get_time_diff, save_checkpoint
from utils.plot import Plotter
from utils.metric import Metric
from utils.transform import *


def train(
    cfg,
    model,
    data_loader,
    optimizer,
    criterion,
    modality,
    logger,
    device=torch.device("cuda"),
):
    no_batches = round(len(data_loader.dataset) / data_loader.batch_size)
    batch_interval = no_batches // 4
    dict_to_device = TransferTensorDict(device)

    model.train()
    train_loss = 0
    for batch_no, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        data, target = dict_to_device(data), dict_to_device(target)

        out = model(data)

        loss = model.get_loss(criterion, target, out)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_no == 0 or (batch_no + 1) % batch_interval == 0:
            logger.info(
                "Batch Progress: [{}/{}] || Train Loss: {:.5f}".format(
                    (batch_no + 1), no_batches, train_loss / (batch_no + 1),
                )
            )

    train_loss /= no_batches
    return train_loss


def validate(
    cfg, model, data_loader, criterion, modality, logger, device=torch.device("cuda")
):
    no_batches = len(data_loader.dataset) // data_loader.batch_size
    dict_to_device = TransferTensorDict(device)
    metric = Metric()

    model.eval()
    val_loss = 0
    val_acc = {}
    precision = {}
    recall = {}
    confusion_matrix = {}
    for cls, no_cls in cfg.MODEL.NUM_CLASSES.items():
        val_acc[cls] = [0] * (len(cfg.VAL.TOPK))
        confusion_matrix[cls] = np.zeros((no_cls, no_cls))
        precision[cls] = 0
        recall[cls] = 0

    with torch.no_grad():
        for data, target, _ in data_loader:
            data, target = dict_to_device(data), dict_to_device(target)

            out = model(data)

            loss = model.get_loss(criterion, target, out)
            val_loss += loss.item()
            for cls in val_acc.keys():
                acc, conf_mat, prec, rec = metric.calculate_metrics(
                    out[cls], target[cls], topk=cfg.VAL.TOPK
                )
                val_acc[cls] = [x + y for x, y in zip(val_acc[cls], acc)]
                precision[cls] += prec
                recall[cls] += rec
                confusion_matrix[cls] += conf_mat

    val_loss /= no_batches
    for cls in val_acc.keys():
        val_acc[cls] = [round(x / no_batches, 2) for x in val_acc[cls]]
        precision[cls] = round(precision[cls] / no_batches, 2)
        recall[cls] = round(recall[cls] / no_batches, 2)

    return val_loss, val_acc, confusion_matrix, precision, recall


def run_trainer(cfg, logger, modality, writer):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = cfg.TRAIN.EPOCHS

    logger.info("Initializing model...")
    model = build_model(cfg, modality)
    logger.info("Model initialized.")
    logger.info("----------------------------------------------------------")

    if cfg.TRAIN.OPTIM.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY,
        )
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg.TRAIN.LR_STEPS, gamma=cfg.TRAIN.LR_DECAY
        )
    elif cfg.TRAIN.OPTIM.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            cfg.TRAIN.LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.TRAIN.WEIGHT_DECAY,
        )
        lr_scheduler = None

    criterion = torch.nn.CrossEntropyLoss()

    if cfg.TRAIN.PRE_TRAINED:
        logger.info("Loading pre-trained weights...")
        data_dict = torch.load(cfg.TRAIN.PRE_TRAINED, map_location="cpu")
        model.load_state_dict(data_dict["model"])
        optimizer.load_state_dict(data_dict["optimizer"])
        start_epoch = data_dict["epoch"] + 1
        logger.info(
            "Model will continue training from epoch no {}".format(start_epoch + 1)
        )
        logger.info("Done.")
        logger.info("----------------------------------------------------------")

    else:
        start_epoch = 0

    checkpoint_name = "{}_{}.pth".format(cfg.MODEL.ARCH, "_".join(modality))
    if cfg.MODEL.CHECKPOINT_PREFIX:
        checkpoint_name = cfg.MODEL.CHECKPOINT_PREFIX + "_" + checkpoint_name
    checkpoint = os.path.join(cfg.DATA.OUT_DIR, "checkpoint", checkpoint_name,)

    model, criterion = model.to(device), criterion.to(device)

    logger.info("Reading list of training and validation videos...")
    with open(cfg.TRAIN.VID_LIST) as f:
        train_list = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

    with open(cfg.VAL.VID_LIST) as f:
        val_list = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

    logger.info("Done.")
    logger.info("----------------------------------------------------------")

    train_transforms = {}
    val_transforms = {}
    for m in modality:
        if m == "RGB":
            train_transforms[m] = torchvision.transforms.Compose(
                [
                    MultiScaleCrop(
                        cfg.DATA.TRAIN_CROP_SIZE, [1, 0.875, 0.75, 0.66], is_flow=False
                    ),
                    RandomHorizontalFlip(prob=0.5, is_flow=False),
                    Stack(m),
                    ToTensor(),
                    Normalize(cfg.DATA.RGB_MEAN, cfg.DATA.RGB_STD),
                ]
            )
            val_transforms[m] = torchvision.transforms.Compose(
                [
                    Rescale(cfg.DATA.TEST_SCALE_SIZE, is_flow=False),
                    CenterCrop(cfg.DATA.TEST_CROP_SIZE),
                    Stack(m),
                    ToTensor(),
                ]
            )
        elif m == "Flow":
            train_transforms[m] = torchvision.transforms.Compose(
                [
                    MultiScaleCrop(
                        cfg.DATA.TRAIN_CROP_SIZE, [1, 0.875, 0.75], is_flow=True
                    ),
                    RandomHorizontalFlip(prob=0.5, is_flow=True),
                    Stack(m),
                    ToTensor(),
                    Normalize(cfg.DATA.FLOW_MEAN, cfg.DATA.FLOW_STD),
                ]
            )
            val_transforms[m] = torchvision.transforms.Compose(
                [
                    Rescale(cfg.DATA.TEST_SCALE_SIZE, is_flow=True),
                    CenterCrop(cfg.DATA.TEST_CROP_SIZE),
                    Stack(m),
                    ToTensor(),
                ]
            )
        elif m == "Audio":
            train_transforms[m] = torchvision.transforms.Compose([Stack(m), ToTensor()])
            val_transforms[m] = torchvision.transforms.Compose([Stack(m), ToTensor()])

    logger.info("Creating datasets...")
    train_dataset = Video_Dataset(
        cfg,
        train_list,
        modality,
        transform=train_transforms,
        mode="train",
        read_pickle=cfg.DATA.READ_AUDIO_PICKLE,
    )

    val_dataset = Video_Dataset(
        cfg,
        val_list,
        modality,
        transform=val_transforms,
        mode="val",
        read_pickle=cfg.DATA.READ_AUDIO_PICKLE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.VAL.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )

    logger.info("Done.")
    logger.info("----------------------------------------------------------")

    min_val_loss = np.inf
    plotter = Plotter(writer)

    logger.info("Training in progress...")
    start_time = time.time()

    for epoch in range(start_epoch, epochs, 1):
        epoch_start_time = time.time()
        train_loss = train(
            cfg, model, train_loader, optimizer, criterion, modality, logger, device
        )

        if lr_scheduler:
            lr_scheduler.step()

        logger.info("Validation in progress...")

        val_loss, val_acc, confusion_matrix, precision, recall = validate(
            cfg, model, val_loader, criterion, modality, logger, device
        )

        if val_loss < min_val_loss:
            save_checkpoint(
                model, optimizer, epoch, confusion_matrix, filename=checkpoint
            )

        hours, minutes, seconds = get_time_diff(epoch_start_time, time.time())

        logger.info("----------------------------------------------------------")
        logger.info(
            "Epoch: [{}/{}] || Train_loss: {:.5f} || Val_Loss: {:.5f}".format(
                epoch + 1, epochs, train_loss, val_loss
            )
        )
        logger.info("----------------------------------------------------------")
        logger.info(
            "Epoch Time: {} hours, {} minutes, {} seconds".format(
                hours, minutes, seconds
            )
        )
        logger.info("----------------------------------------------------------")
        logger.info("Accuracy Top {}:".format(cfg.VAL.TOPK))
        logger.info(json.dumps(val_acc, indent=2))
        logger.info("Precision: {}".format(json.dumps(precision, indent=2)))
        logger.info("Recall: {}".format(json.dumps(recall, indent=2)))
        logger.info("----------------------------------------------------------")

        plotter.plot_scalar(train_loss, epoch, "train/loss")
        plotter.plot_scalar(val_loss, epoch, "val/loss")
        plotter.plot_dict(precision, epoch, "val/precision")
        plotter.plot_dict(recall, epoch, "val/recall")
        plotter.plot_dict(val_acc, epoch, "val/accuracy/top")

    hours, minutes, seconds = get_time_diff(start_time, time.time())
    logger.info(
        "Training completed. Total time taken: {} hours, {} minutes, {} seconds".format(
            hours, minutes, seconds
        )
    )

