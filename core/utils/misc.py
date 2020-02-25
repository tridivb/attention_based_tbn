import torch
import json
import numpy as np
import os


def get_modality(cfg):
    """
    Helper function to intialize list of modalities

    Args
    ----------
    cfg: dict
        Dictionary of config parameters
    """

    modality = []

    if cfg.data.rgb.enable:
        modality.append("RGB")
    if cfg.data.flow.enable:
        modality.append("Flow")
    if cfg.data.audio.enable:
        modality.append("Audio")

    return modality


def get_time_diff(start_time, end_time):
    """
    Helper function to calculate time difference

    Args
    ----------
    start_time: float
        Start time in seconds since January 1, 1970, 00:00:00 (UTC)
    end_time: float
        End time in seconds since January 1, 1970, 00:00:00 (UTC)

    Returns
    ----------
    hours: int
        Difference of hours between start and end time
    minutes: int
        Difference of minutes between start and end time
    seconds: int
        Difference of seconds between start and end time
    """

    hours = int((end_time - start_time) / 3600)
    minutes = int((end_time - start_time) / 60) - (hours * 60)
    seconds = int(np.floor((end_time - start_time) % 60))
    return (hours, minutes, seconds)


def save_checkpoint(
    model,
    optimizer,
    epoch,
    train_loss_hist,
    val_loss_hist,
    val_acc_hist,
    confusion_matrix,
    num_gpus,
    scheduler=None,
    filename="checkpoint.pth",
):
    """
    Helper function to save model checkpoint

    Args
    ----------
    model: torch.nn.model
        Trained model
    optimizer: optim
        Trained optimizer
    epoch: int
        Epoch progress till the point of saving
    train_loss_hist: list
        History of train losses
    val_loss_hist: list
        History of validation losses
    val_loss_acc: list
        History of validation accuracies
    confusion_matrix: np.ndarray
        Confusion matrix for the final epoch over the validation/test set
    scheduler: torch.optim.lr_scheduler, default = None
        Trained Learning rate scheduler
    filename: str, default = "checkpoint.pth"
        Checkpoint file name
    """

    data = {
        "epoch": epoch,
        "train_loss": train_loss_hist,
        "validation_loss": val_loss_hist,
        "validation_accuracy": val_acc_hist,
        "optimizer": optimizer.state_dict(),
    }

    if num_gpus > 1:
        data["model"] = model.module.state_dict()
    else:
        data["model"] = model.state_dict()

    if confusion_matrix:
        data["conf_mat"] = confusion_matrix

    if scheduler:
        data["scheduler"] = scheduler.state_dict()

    torch.save(data, filename)


def save_scores(scores, file_name):
    """
    Helper function to save output prediction scores for epic kitchens

    Args
    ----------
    scores: dict
        Dictionary of model outputs from forward pass over test set
    file_name: str
        Output file name
    """

    out_result = {}
    out_result["version"] = "0.1"
    out_result["challenge"] = "action_recognition"

    for key in scores.keys():
        scores[key] = torch.cat(scores[key], dim=0)
        # if key != "action_id":
        #     scores[key] = torch.nn.functional.softmax(scores[key], dim=1)

    results = {}

    no_of_ids = scores["action_id"].shape[0]

    for idx in range(no_of_ids):
        a_id = str(scores["action_id"][idx].item())
        results[a_id] = {}
        for key in scores.keys():
            if key != "action_id":
                results[a_id][key] = {
                    str(id): s.item() for id, s in enumerate(scores[key][idx])
                }

    out_result["results"] = results

    os.makedirs(os.path.split(file_name)[0], exist_ok=True)

    with open(file_name, "w") as f:
        json.dump(out_result, f, indent=4)
