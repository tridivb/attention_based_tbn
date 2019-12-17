#!/usr/bin/env python

import os
import time
import json
import numpy as np
import torch
import torch.optim as optim
import torchvision
from torch.utils.data.dataloader import DataLoader

from models.model_builder import build_model
from dataset.dataset import Video_Dataset
from utils.misc import get_time_diff, save_checkpoint
from utils.plot import Plotter
from utils.metric import Metric
from dataset.transform import *


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

        if cfg.TRAIN.CLIP_GRAD:
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.TRAIN.CLIP_GRAD
            )
            if total_norm > cfg.TRAIN.CLIP_GRAD:
                logger.info(
                    "Clipping gradient: {} with coef {}".format(
                        total_norm, cfg.TRAIN.CLIP_GRAD / total_norm
                    )
                )

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
    confusion_matrix = {}
    for cls, no_cls in cfg.MODEL.NUM_CLASSES.items():
        val_acc[cls] = [0] * (len(cfg.VAL.TOPK))
        confusion_matrix[cls] = torch.zeros((no_cls, no_cls), device=device)

    with torch.no_grad():
        for data, target, _ in data_loader:
            data, target = dict_to_device(data), dict_to_device(target)

            out = model(data)

            loss = model.get_loss(criterion, target, out)
            val_loss += loss.item()
            for cls in val_acc.keys():
                acc, conf_mat = metric.calculate_metrics(
                    out[cls], target[cls], device, topk=cfg.VAL.TOPK
                )
                val_acc[cls] = [x + y for x, y in zip(val_acc[cls], acc)]
                confusion_matrix[cls] += conf_mat

    val_loss /= no_batches
    for cls in val_acc.keys():
        val_acc[cls] = [round(x / no_batches, 2) for x in val_acc[cls]]
        if device.type == "cuda":
            confusion_matrix[cls] = confusion_matrix[cls].cpu()
        confusion_matrix[cls] = confusion_matrix[cls].numpy()

    return (
        val_loss,
        val_acc,
        confusion_matrix,
    )


def run_trainer(cfg, logger, modality, writer):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = cfg.TRAIN.EPOCHS

    logger.info("Initializing model...")
    model, criterion = build_model(cfg, modality, device)
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

    if cfg.TRAIN.PRE_TRAINED:
        logger.info("Loading pre-trained weights...")
        data_dict = torch.load(cfg.TRAIN.PRE_TRAINED)
        if cfg.NUM_GPUS > 1:
            model.module.load_state_dict(data_dict["model"])
        else:
            model.load_state_dict(data_dict["model"])
        optimizer.load_state_dict(data_dict["optimizer"])
        if lr_scheduler and "scheduler" in data_dict.keys():
            lr_scheduler.load_state_dict(data_dict["scheduler"])
        start_epoch = data_dict["epoch"] + 1
        epochs += start_epoch
        train_loss_hist = data_dict["train_loss"]
        val_loss_hist = data_dict["validation_loss"]
        val_acc_hist = data_dict["validation_accuracy"]
        logger.info(
            "Model will continue training from epoch no {}".format(start_epoch + 1)
        )
        logger.info("Done.")
        logger.info("----------------------------------------------------------")

    else:
        start_epoch = 0
        train_loss_hist = []
        val_loss_hist = []
        val_acc_hist = {k: [] for k in cfg.MODEL.NUM_CLASSES.keys()}

    checkpoint_name = "tbn_{}_{}.pth".format(cfg.MODEL.ARCH, "_".join(modality))
    if cfg.DATA.DATASET:
        checkpoint_name = "_".join([cfg.DATA.DATASET, checkpoint_name])
    checkpoint = os.path.join(cfg.MODEL.CHECKPOINT_DIR, checkpoint_name)

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
        cfg.TRAIN.ANNOTATION_FILE,
        modality,
        transform=train_transforms,
        mode="train",
    )

    val_dataset = Video_Dataset(
        cfg,
        val_list,
        cfg.TRAIN.ANNOTATION_FILE,
        modality,
        transform=val_transforms,
        mode="val",
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

        train_loss_hist.append(train_loss)
        logger.info("Validation in progress...")

        if cfg.VAL.VAL_ENABLE:
            val_loss, val_acc, confusion_matrix = validate(
                cfg, model, val_loader, criterion, modality, logger, device
            )
            val_loss_hist.append(val_loss)
            for k in val_acc_hist.keys():
                val_acc_hist[k].append(val_acc[k])
        else:
            val_loss = 0

        if lr_scheduler:
            lr_scheduler.step()

        if val_loss < min_val_loss or not cfg.VAL.VAL_ENABLE:
            if cfg.NUM_GPUS > 1:
                save_checkpoint(
                    model.module,
                    optimizer,
                    epoch,
                    train_loss_hist,
                    val_loss_hist,
                    val_acc_hist,
                    confusion_matrix,
                    scheduler=lr_scheduler,
                    filename=os.path.splitext(checkpoint)[0] + "_best.pth",
                )
            else:
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    train_loss_hist,
                    val_loss_hist,
                    val_acc_hist,
                    confusion_matrix,
                    scheduler=lr_scheduler,
                    filename=os.path.splitext(checkpoint)[0] + "_best.pth",
                )
            min_val_loss = val_loss

        save_checkpoint(
            model,
            optimizer,
            epoch,
            train_loss_hist,
            val_loss_hist,
            val_acc_hist,
            confusion_matrix,
            scheduler=lr_scheduler,
            filename=checkpoint,
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
        logger.info("----------------------------------------------------------")

        plotter.plot_scalar(train_loss, epoch, "train/loss")
        plotter.plot_scalar(val_loss, epoch, "val/loss")
        for cls, acc in val_acc.items():
            for k, v in enumerate(acc):
                plotter.plot_scalar(
                    v, epoch, "val/accuracy/{}_top_{}".format(cls, cfg.VAL.TOPK[k])
                )

    hours, minutes, seconds = get_time_diff(start_time, time.time())
    logger.info(
        "Training completed. Total time taken: {} hours, {} minutes, {} seconds".format(
            hours, minutes, seconds
        )
    )

