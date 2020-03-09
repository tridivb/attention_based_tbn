#!/usr/bin/env python

import os
import time
import json
import numpy as np
import torch
import torch.optim as optim
import torchvision
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from collections import OrderedDict

from core.models import build_model
from core.dataset import Video_Dataset
from core.utils import get_time_diff, save_checkpoint, Plotter, Metric
from core.dataset.transform import *


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
    """
    Train the model

    Args
    ----------
    cfg: dict
        Dictionary of config parameters
    model: torch.nn.model
        Model to train
    data_loader: DataLoader
        Data loader to iterate over the data
    optimizer: optim
        Optimizer to use
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
    train_loss: float
        Overall training loss over an epoch

    """
    no_batches = round(len(data_loader.dataset) / data_loader.batch_size)
    batch_interval = no_batches // 4
    dict_to_device = TransferTensorDict(device)
    metric = Metric(cfg, no_batches, device)
    loss_tracker = 0

    model.train()
    for batch_no, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        data, target = dict_to_device(data), dict_to_device(target)

        out = model(data)

        loss, batch_size = model.get_loss(criterion, target, out)
        metric.set_metrics(out, target, batch_size, loss)
        loss["total"].backward()
        loss_tracker += loss["total"].item()

        if cfg.train.clip_grad:
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.train.clip_grad
            )
            if total_norm > cfg.train.clip_grad:
                logger.info(
                    f"Clipping gradient: {total_norm} with coef {cfg.train.clip_grad / total_norm}"
                )

        optimizer.step()

        if batch_no == 0 or (batch_no + 1) % batch_interval == 0:
            logger.info(
                "Batch Progress: [{}/{}] || Train Loss: {:.5f}".format(
                    (batch_no + 1), no_batches, loss_tracker / (batch_no + 1),
                )
            )

    train_loss, _, _ = metric.get_metrics()
    return train_loss


def validate(
    cfg, model, data_loader, criterion, modality, logger, device=torch.device("cuda")
):
    """
    Validate the model

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
    val_loss,: float
        Overall validation loss
    val_acc: dict
        Accuracy of each type of class
    confusion_matrix: Tensor
        Array of the confusion matrix over the validation set

    """

    no_batches = len(data_loader.dataset) // data_loader.batch_size
    dict_to_device = TransferTensorDict(device)
    metric = Metric(cfg, no_batches, device)

    model.eval()

    with torch.no_grad():
        for data, target, _ in tqdm(data_loader):
            data, target = dict_to_device(data), dict_to_device(target)

            out = model(data)

            loss, batch_size = model.get_loss(criterion, target, out)
            metric.set_metrics(out, target, batch_size, loss)

    val_loss, val_acc, conf_mat = metric.get_metrics()

    return (val_loss, val_acc, conf_mat)


def run_trainer(cfg, logger, modality, writer):
    """
    Initialize model , data loaders, loss function, optimizer and execute the training

    Args
    ----------
    cfg: dict
        Dictionary of config parameters
    logger: logger
        Python logger
    modality: list
        List of input modalities
    writer: SummaryWriter
        Tensorboard writer to plot the metrics during training

    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = cfg.train.epochs

    logger.info("Initializing model...")
    model, criterion, num_gpus = build_model(cfg, modality, device)
    logger.info("Model initialized.")
    logger.info(model)
    logger.info("----------------------------------------------------------")

    if cfg.train.optim.type.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            cfg.train.optim.lr,
            momentum=cfg.train.optim.momentum,
            weight_decay=cfg.train.optim.weight_decay,
        )
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.train.scheduler.lr_steps,
            gamma=cfg.train.scheduler.lr_decay,
        )
    elif cfg.train.optim.type.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            cfg.train.optim.lr,
            betas=(0.9, 0.999),
            weight_decay=cfg.train.optim.weight_decay,
        )
        lr_scheduler = None

    if cfg.train.pre_trained:
        logger.info("Loading pre-trained weights...")
        data_dict = torch.load(cfg.train.pre_trained)
        if num_gpus > 1:
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
        logger.info(f"Model will continue training from epoch no {start_epoch + 1}")
        logger.info("Done.")
        logger.info("----------------------------------------------------------")

    else:
        start_epoch = 0
        train_loss_hist = []
        val_loss_hist = []
        val_acc_hist = {k: [] for k in cfg.model.num_classes.keys()}

    checkpoint_name = "tbn_{}_{}.pth".format(cfg.model.arch, "_".join(modality))
    if cfg.data.dataset:
        checkpoint_name = "_".join([cfg.data.dataset, checkpoint_name])
    checkpoint = os.path.join(
        cfg.out_dir, cfg.model.checkpoint_dir, cfg.exp_name, checkpoint_name
    )
    os.makedirs(os.path.split(checkpoint)[0], exist_ok=True)

    logger.info("Reading list of training and validation videos...")
    file_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    with open(os.path.join(file_dir, cfg.train.vid_list)) as f:
        train_list = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

    with open(os.path.join(file_dir, cfg.val.vid_list)) as f:
        val_list = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

    logger.info("Done.")
    logger.info("----------------------------------------------------------")

    train_transforms = OrderedDict()
    val_transforms = OrderedDict()
    for m in modality:
        if m == "RGB":
            train_transforms[m] = torchvision.transforms.Compose(
                [
                    MultiScaleCrop(cfg.data.train_crop_size, [1, 0.875, 0.75, 0.66]),
                    RandomHorizontalFlip(prob=0.5),
                    Stack(m),
                    ToTensor(),
                    Normalize(cfg.data.rgb.mean, cfg.data.rgb.std),
                ]
            )
            val_transforms[m] = torchvision.transforms.Compose(
                [
                    Rescale(cfg.data.test_scale_size),
                    CenterCrop(cfg.data.test_crop_size),
                    Stack(m),
                    ToTensor(),
                    Normalize(cfg.data.rgb.mean, cfg.data.rgb.std),
                ]
            )
        elif m == "Flow":
            train_transforms[m] = torchvision.transforms.Compose(
                [
                    MultiScaleCrop(cfg.data.train_crop_size, [1, 0.875, 0.75]),
                    RandomHorizontalFlip(prob=0.5),
                    Stack(m),
                    ToTensor(),
                    Normalize(cfg.data.flow.mean, cfg.data.flow.std),
                ]
            )
            val_transforms[m] = torchvision.transforms.Compose(
                [
                    Rescale(cfg.data.test_scale_size),
                    CenterCrop(cfg.data.test_crop_size),
                    Stack(m),
                    ToTensor(),
                    Normalize(cfg.data.flow.mean, cfg.data.flow.std),
                ]
            )
        elif m == "Audio":
            train_transforms[m] = torchvision.transforms.Compose(
                [Stack(m), ToTensor(is_audio=True)]
            )
            val_transforms[m] = torchvision.transforms.Compose(
                [Stack(m), ToTensor(is_audio=True)]
            )

    logger.info("Creating datasets...")
    train_dataset = Video_Dataset(
        cfg,
        train_list,
        cfg.train.annotation_file,
        modality,
        transform=train_transforms,
        mode="train",
    )

    val_dataset = Video_Dataset(
        cfg,
        val_list,
        cfg.train.annotation_file,
        modality,
        transform=val_transforms,
        mode="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.val.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    logger.info("Done.")
    logger.info("----------------------------------------------------------")

    best_acc = np.NINF
    plotter = Plotter(writer)
    plotter.add_config(cfg)

    logger.info("Training in progress...")
    start_time = time.time()

    for epoch in range(start_epoch, epochs, 1):
        epoch_start_time = time.time()
        train_loss = train(
            cfg, model, train_loader, optimizer, criterion, modality, logger, device
        )

        train_loss_hist.append(train_loss)
        logger.info("Validation in progress...")

        if cfg.val.enable:
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

        if cfg.val.enable and val_acc["all_class"][0] > best_acc:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                train_loss_hist,
                val_loss_hist,
                val_acc_hist,
                confusion_matrix,
                num_gpus,
                scheduler=lr_scheduler,
                filename=os.path.splitext(checkpoint)[0] + "_best.pth",
            )
            best_acc = val_acc["all_class"][0]

        save_checkpoint(
            model,
            optimizer,
            epoch,
            train_loss_hist,
            val_loss_hist,
            val_acc_hist,
            confusion_matrix,
            num_gpus,
            scheduler=lr_scheduler,
            filename=checkpoint,
        )

        hours, minutes, seconds = get_time_diff(epoch_start_time, time.time())

        logger.info("----------------------------------------------------------")
        logger.info(f"Epoch: [{epoch + 1}/{epochs}]")
        logger.info(f"Train_loss: {train_loss}")
        logger.info(f"Val_Loss: {val_loss}")
        logger.info("----------------------------------------------------------")
        logger.info(f"Epoch Time: {hours} hours, {minutes} minutes, {seconds} seconds")
        logger.info("----------------------------------------------------------")
        logger.info(f"Accuracy Top {cfg.val.topk}:")
        logger.info(json.dumps(val_acc, indent=2))
        logger.info("----------------------------------------------------------")

        for k in train_loss.keys():
            plotter.plot_scalar(train_loss[k], epoch, f"train/{k}_loss")
            plotter.plot_scalar(val_loss[k], epoch, f"val/{k}_loss")
        for cls, acc in val_acc.items():
            for k, v in enumerate(acc):
                plotter.plot_scalar(
                    v, epoch, f"val/accuracy/{cls}_top_{cfg.val.topk[k]}"
                )

    hours, minutes, seconds = get_time_diff(start_time, time.time())
    logger.info(
        f"Training completed. Total time taken: {hours} hours, {minutes} minutes, {seconds} seconds"
    )

