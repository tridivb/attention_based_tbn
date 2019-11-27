import torch
import torchvision
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import numpy as np
import os
from tensorboardX import SummaryWriter

from models.model_builder import build_model
from utils.dataset import Video_Dataset
from utils.misc import *
from utils.transform import *


def train(cfg, logger, modality):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    epochs = cfg.TRAIN.EPOCHS

    model = build_model(cfg, modality)

    if cfg.MODEL.CHECKPOINT:
        model.load_state_dict(torch.load(cfg.MODEL.CHECKPOINT))

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

    model, criterion = model.to(device), criterion.to(device)

    with open(cfg.TRAIN.VID_LIST) as f:
        train_list = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

    with open(cfg.TEST.VID_LIST) as f:
        val_list = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

    train_transforms = {}
    val_transforms = {}
    for m in modality:
        if m == "RGB":
            train_transforms[m] = torchvision.transforms.Compose(
                [
                    MultiScaleCrop(cfg.DATA.TRAIN_CROP_SIZE, [1, 0.875, 0.75, 0.66]),
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
                    Normalize(cfg.DATA.RGB_MEAN, cfg.DATA.RGB_STD),
                ]
            )
        elif m == "Flow":
            train_transforms[m] = torchvision.transforms.Compose(
                [
                    MultiScaleCrop(
                        cfg.DATA.TRAIN_CROP_SIZE, [1, 0.875, 0.75, 0.66], is_flow=True
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

    train_dataset = Video_Dataset(
        cfg, train_list, modality, transform=train_transforms, mode="train"
    )

    val_dataset = Video_Dataset(
        cfg, train_list, modality, transform=val_transforms, mode="val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_no, (input) in enumerate(train_loader):
            optimizer.zero_grad()
            target = input["target"]
            for m in modality:
                input[m] = input[m].to(device)
            for key in target.keys():
                target[key] = target[key].to(device)
            out = model(input, num_segments = cfg.DATA.NUM_SEGMENTS)

            loss = calculate_loss(criterion, target, out)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            print(loss.item(), train_loss)

        # model.eval()
        # val_loss = 0
        # val_acc = 0
        # with torch.no_grad():
        #     for input in val_loader:
        #         frames, target = input["frames"].to(device), input["target"].to(device)
        #         out = model(frames)
        #         loss = calculate_loss(criterion, target, out)
        #         val_loss += loss.item()
        #         val_acc += calculate_topk_accuracy(out, target, topk=[1, 5])

    hours, minutes, seconds = get_time_diff(start_time, time.time())
