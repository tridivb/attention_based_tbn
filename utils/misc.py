import torch
import torchvision
import numpy as np
import time
from collections import OrderedDict

def calculate_loss(criterion, target, preds):
    assert (target, isinstance(OrderedDict))
    assert (preds, isinstance(OrderedDict))

    loss = 0

    for key in target.keys():
        loss += criterion(preds[key], target[key])

    return loss

def calculate_topk_accuracy(out, target, topk=[1,]):
    maxk = max(topk)
    batch_size = target.size(0)

    _, preds = out.topk(maxk, 1, True, True)
    preds = preds.t()
    correct = preds.eq(target.view(1, -1).expand_as(preds))

    acc = {}
    for k in topk:
        correct_k = correct[:k].view(-1).to(torch.float32).sum(0)
        acc["top_{}".format(k)] = (float(correct_k.mul_(100.0 / batch_size)))
    return acc

def get_time_diff(start_time, end_time):
    hours = int((end_time-start_time)/3600)
    minutes = int((end_time-start_time)/60)-(hours*60)
    seconds = int((end_time-start_time) % 60)
    return (hours, minutes, seconds)

def save_checkpoint(model, optimizer, epoch):
    raise Exception("Not implemented error")