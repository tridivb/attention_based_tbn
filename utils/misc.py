import torch
import numpy as np


def get_time_diff(start_time, end_time):
    hours = int((end_time - start_time) / 3600)
    minutes = int((end_time - start_time) / 60) - (hours * 60)
    seconds = int((end_time - start_time) % 60)
    return (hours, minutes, seconds)


def save_checkpoint(model, optimizer, epoch, scheduler=None, filename="checkpoint.pth"):
    data = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if scheduler:
        data["scheduler"] = scheduler.state_dict()

    torch.save(data, filename)
