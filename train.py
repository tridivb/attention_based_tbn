import os
import time
import numpy as np
import torch
import torch.optim as optim
import torchvision
from torch.utils.data.dataloader import DataLoader
from tensorboardX import SummaryWriter

from models.model_builder import build_model
from utils.dataset import Video_Dataset
from utils.misc import get_time_diff, calculate_topk_accuracy, save_checkpoint
from utils.transform import *


def train(cfg, logger, modality):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    epochs = cfg.TRAIN.EPOCHS

    print("Initializing model...")
    model = build_model(cfg, modality)
    print("Model initialized.")
    print("----------------------------------------------------------")

    if cfg.MODEL.CHECKPOINT:
        print("Loading pre-trained weights...")
        model.load_state_dict(torch.load(cfg.MODEL.CHECKPOINT))
        print("Done.")
        print("----------------------------------------------------------")
    
    checkpoint_name = os.path.join("./weights", "model_{}_{}.pth".format(cfg.MODEL.ARCH, "_".join(modality)))
    

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

    print("Creating list of training and validation videos...")
    with open(cfg.TRAIN.VID_LIST) as f:
        train_list = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

    with open(cfg.VAL.VID_LIST) as f:
        val_list = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

    print("Done.")
    print("----------------------------------------------------------")

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
                    Normalize(cfg.DATA.FLOW_MEAN, cfg.DATA.FLOW_STD),
                ]
            )
        elif m == "Audio":
            train_transforms[m] = torchvision.transforms.Compose([Stack(m), ToTensor()])
            val_transforms[m] = torchvision.transforms.Compose([Stack(m), ToTensor()])

    print("Creating datasets...")
    train_dataset = Video_Dataset(
        cfg, train_list, modality, transform=train_transforms, mode="train"
    )

    val_dataset = Video_Dataset(
        cfg, val_list, modality, transform=val_transforms, mode="val"
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

    print("Done.")
    print("----------------------------------------------------------")

    no_train_batches = len(train_loader.dataset) // train_loader.batch_size
    no_val_batches = len(val_loader.dataset) // val_loader.batch_size

    # batch_interval = int((len(train_loader.dataset) / train_loader.batch_size) // 4)
    batch_interval = 1

    dict_to_device = TransferTensorDict(device)

    min_val_loss = np.inf

    print("Training in progress...")
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        for batch_no, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data, target = dict_to_device(data), dict_to_device(target)

            out = model(data)

            loss = model.get_loss(criterion, target, out)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            if batch_no == 0 or (batch_no + 1) % batch_interval == 0:
                print(
                    "Epoch Progress: [{}/{}] || Train Loss: {:.5f}".format(
                        (batch_no + 1) * train_loader.batch_size,
                        len(train_loader.dataset),
                        train_loss / (batch_no + 1),
                    )
                )

        if lr_scheduler:
            lr_scheduler.step()

        train_loss /= no_train_batches

        model.eval()
        val_loss = 0
        val_acc = {}
        for cls in cfg.MODEL.NUM_CLASSES:
            val_acc[cls] = [0] * (len(cfg.VAL.TOPK))

        with torch.no_grad():
            for data, target in val_loader:
                data, target = dict_to_device(data), dict_to_device(target)

                out = model(data)

                loss = model.get_loss(criterion, target, out)
                val_loss += loss.item()
                for cls in val_acc.keys():
                    acc = calculate_topk_accuracy(
                        out[cls], target[cls], topk=cfg.VAL.TOPK
                    )
                    val_acc[cls] = [x + y for x, y in zip(val_acc[cls], acc)]

        val_loss /= no_val_batches
        for cls in val_acc.keys():
            val_acc[cls] = [x / no_val_batches for x in val_acc[cls]]

        if val_loss < min_val_loss:
            save_checkpoint(model, optimizer, epoch, filename=checkpoint_name)

        hours, minutes, seconds = get_time_diff(epoch_start_time, time.time())
        print("----------------------------------------------------------")
        print(
            "Epoch: [{}/{}] || Train_loss: {:.5f} || Val_Loss: {:.5f}".format(
                epoch + 1, epochs, train_loss, val_loss
            )
        )
        print("----------------------------------------------------------")
        print("Validation Accuracy (Top {}): {}".format(cfg.VAL.TOPK, val_acc))
        print("----------------------------------------------------------")
        print(
            "Epoch Time: {} hours, {} minutes, {} seconds,".format(
                hours, minutes, seconds
            )
        )
        print("----------------------------------------------------------")

    hours, minutes, seconds = get_time_diff(start_time, time.time())
    print(
        "Training completed. Total time taken: {} hours, {} minutes, {} seconds,".format(
            hours, minutes, seconds
        )
    )

