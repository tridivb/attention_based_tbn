import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import numpy as np
import os
from tensorboardX import SummaryWriter

from models.model_builder import build_model
from utils.dataset import Video_Dataset
from utils.misc import *


def test(cfg, logger):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
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

    criterion = torch.nn.CrossEntropyLoss()

    model, criterion = model.to(device), criterion.to(device)

    # TODO - Read video list file
    vid_list = None

    dataset = Video_Dataset(cfg, vid_list, modality, mode="train")

    test_loader = DataLoader(dataset, num_workers=cfg.NUM_WORKERS)

    start_time = time.time()

    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for input in test_loader:
            frames, target = input["frames"].to(device), input["target"].to(device)
            out = model(frames)
            loss = calculate_loss(criterion, target, out)
            test_loss += loss.item()
            test_acc += calculate_topk_accuracy(out, target, topk=[1, 5])

    hours, minutes, seconds = get_time_diff(start_time, time.time())