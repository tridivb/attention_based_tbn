import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import numpy as np
import os
from tensorboardX import SummaryWriter

from models.model_builder import build_model
from utils.dataset import Video_Dataset
from utils.misc import *


def train(cfg, logger):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    epochs = cfg.TRAIN.EPOCHS
    
    modality = []

    if cfg.DATA.USE_RGB:
        modality.append("RGB")
    if cfg.DATA.USE_FLOW:
        modality.append("Flow")
    if cfg.DATA.USE_RGB:
        modality.append("Audio")
    
    model = build_model(cfg)

    if cfg.MODEL.CHECKPOINT:
        model.load_state_dict(torch.load(cfg.MODEL.CHECKPOINT))

    if cfg.TRAIN.OPTIM.tolower() == "sgd":
        optimizer = optim.SGD(model.parameters(), cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, cfg.TRAIN.LR_DECAY, gamma=cfg.TRAIN.GAMMA)
    elif cfg.TRAIN.OPTIM.tolower() == "adam":
        optimizer = optim.Adam(model.parameters(), cfg.TRAIN.LR, betas=(0.9, 0.999), weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        lr_scheduler = None

    criterion = torch.nn.CrossEntropyLoss()

    model, criterion = model.to(device), criterion.to(device)

    # TODO - Read video list file
    vid_list = None

    dataset = Video_Dataset(cfg, vid_list, modality, mode="train")
    
    # TODO - Create train and validation split

    train_loader = DataLoader(dataset, num_workers=cfg.NUM_WORKERS)
    val_loader = DataLoader(dataset, num_workers=cfg.NUM_WORKERS)

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_no, input in train_loader:
            optimizer.zero_grad()
            frames, target = input["frames"].to(device), input["target"].to(device)
            out = model(frames)

            loss = calculate_loss(criterion, target, out)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for input in val_loader:
                frames, target = input["frames"].to(device), input["target"].to(device)
                out = model(frames)
                loss = calculate_loss(criterion, target, out)
                val_loss += loss.item()
                val_acc += calculate_topk_accuracy(out, target, topk=[1, 5])

    hours, minutes, seconds = get_time_diff(start_time, time.time())