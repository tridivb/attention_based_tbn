import os
import time
import numpy as np
import torch
import torchvision
from torch.utils.data.dataloader import DataLoader
from tensorboardX import SummaryWriter

from models.model_builder import build_model
from utils.dataset import Video_Dataset
from utils.misc import get_time_diff, calculate_topk_accuracy
from utils.transform import *


def test(cfg, logger, modality):

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
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
    )

    no_test_batches = len(test_loader.dataset) // test_loader.batch_size

    dict_to_device = TransferTensorDict(device)

    start_time = time.time()

    model.eval()
    test_loss = 0
    test_acc = {}
    for cls in cfg.MODEL.NUM_CLASSES:
        test_acc[cls] = [0] * (len(cfg.TEST.TOPK))

    with torch.no_grad():
        for input, target in test_loader:
            input, target = dict_to_device(input), dict_to_device(target)
            for m in modality:
                b, n, c, h, w = input[m].shape
                input[m] = input[m].view(b * n, c, h, w)

            for cls in input["target"].keys():
                target[cls] = target[cls].repeat(cfg.DATA.NUM_SEGMENTS).to(device)

            out = model(input)
            loss = model.get_loss(criterion, target, out)
            test_loss += loss.item()
            for cls in test_acc.keys():
                acc = calculate_topk_accuracy(out[cls], target[cls], topk=cfg.TEST.TOPK)
                test_acc[cls] = [x + y for x, y in zip(test_acc[cls], acc)]

    test_loss /= no_test_batches
    for cls in test_acc.keys():
        test_acc[cls] = [x / no_test_batches for x in test_acc[cls]]

    print("----------------------------------------------------------")
    print("Test_Loss: {:5f}".format(test_loss))
    print("----------------------------------------------------------")
    print("Validation Accuracy (Top {}): {}".format(cfg.TEST.TOPK, val_acc))
    print("----------------------------------------------------------")

    hours, minutes, seconds = get_time_diff(start_time, time.time())
    print(
        "Inference time: {} hours, {} minutes, {} seconds,".format(
            hours, minutes, seconds
        )
    )
