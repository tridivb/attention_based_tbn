import torch
import json
import numpy as np


def get_modality(cfg):
    modality = []

    if cfg.DATA.USE_RGB:
        modality.append("RGB")
    if cfg.DATA.USE_FLOW:
        modality.append("Flow")
    if cfg.DATA.USE_AUDIO:
        modality.append("Audio")

    return modality


def get_time_diff(start_time, end_time):
    hours = int((end_time - start_time) / 3600)
    minutes = int((end_time - start_time) / 60) - (hours * 60)
    seconds = int((end_time - start_time) % 60)
    return (hours, minutes, seconds)


def save_checkpoint(
    model,
    optimizer,
    epoch,
    train_loss_hist,
    val_loss_hist,
    val_acc_hist,
    confusion_matrix,
    scheduler=None,
    filename="checkpoint.pth",
):
    data = {
        "epoch": epoch,
        "train_loss": train_loss_hist,
        "validation_loss": val_loss_hist,
        "validation_accuracy": val_acc_hist,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if confusion_matrix:
        data["conf_mat"] = confusion_matrix

    if scheduler:
        data["scheduler"] = scheduler.state_dict()

    torch.save(data, filename)


def save_scores(scores, file_name):

    out_result = {}
    out_result["version"] = 0.1
    out_result["challenge"] = "action_recognition"

    for key in scores.keys():
        scores[key] = np.concatenate(scores[key], axis=0)

    results = {}

    no_of_ids = scores["action_id"].shape[0]

    for idx in range(no_of_ids):
        a_id = str(scores["action_id"][idx])
        results[a_id] = {}
        for cls in scores.keys():
            if cls != "action_id":
                results[a_id][cls] = {
                    str(id): score.item() for id, score in enumerate(scores[cls][idx])
                }

    out_result["results"] = results

    with open(file_name, "w") as f:
        json.dump(out_result, f, indent=4)
