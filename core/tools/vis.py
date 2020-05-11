#!/usr/bin/env python

import os
import datetime
import numpy as np
import torch
import torchvision
import pandas as pd
from torch.utils.data.dataloader import default_collate
from omegaconf import OmegaConf
from PIL import Image
import matplotlib.pyplot as plt
import moviepy.editor as mpe
from tqdm import tqdm

from core.models import build_model
from core.dataset import Video_Dataset, EpicClasses
from core.utils import get_modality
from core.dataset.transform import *


def get_interest_points(model, dataset, device, topk=5):
    losses = []
    #     criterion = torch.nn.MSELoss()
    dict_to_device = TransferTensorDict(device)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=8,
    )
    model.eval()
    with torch.no_grad():
        for data, target, _ in tqdm(data_loader):
            data, target = dict_to_device(data), dict_to_device(target)
            out = model(data)
            if "weights" in out.keys():
                b = 1
                n, _, _ = out["weights"].shape
                #             loss = criterion(
                #                 out["weights"].view(b * n, -1), target["weights"].view(b * n, -1)
                #             )
                weights = out["weights"].view(b * n, -1)
                entropy = (-1 * (weights * torch.log(weights + 1e-6)).sum(1)).mean()
                losses.append(entropy.item())
            else:
                raise Exception(
                    "No attention weights found in model output. Please check if model initilization was correct."
                )

    interest_pts = np.array(losses).argsort()[::-1] + 1
    return interest_pts[:topk]


def save_action_segment(data_dir, vid_id, start_time, stop_time):
    vid_file = os.path.join(data_dir, f"vid_symlinks/{vid_id}.MP4")
    video = mpe.VideoFileClip(vid_file).subclip(start_time, stop_time)
    video.write_videofile("results/temp.MP4", logger=None)
    video.close()


def visualize(cfg, model, dataset, index, epic_classes, device):
    dict_to_device = TransferTensorDict(device)
    data, target, _ = default_collate([dataset[index - 1]])
    rgb_indices = data["indices"]["RGB"].numpy().squeeze()
    spec = data["Audio"]
    data = dict_to_device(data)
    model.eval()
    with torch.no_grad():
        out = model(data)

    verb_preds = out["verb"].softmax(dim=1).topk(5, 1, largest=True, sorted=True)
    noun_preds = out["noun"].softmax(dim=1).topk(5, 1, largest=True, sorted=True)
    verb_t5 = [epic_classes.verbs[i.item()] for i in verb_preds.indices.squeeze()]
    noun_t5 = [epic_classes.nouns[i.item()] for i in noun_preds.indices.squeeze()]
    if target["class"]["verb"].item() == -1 or target["class"]["noun"].item() == -1:
        verb_gt = "Unknown"
        noun_gt = "Unknown"
        verb_scores = 0.0
        noun_scores = 0.0
    else:
        verb_gt = epic_classes.verbs[target["class"]["verb"].item()]
        noun_gt = epic_classes.nouns[target["class"]["noun"].item()]
        verb_scores = 1.0
        noun_scores = 1.0
    verbs = [verb_gt] + verb_t5
    verb_scores = [verb_scores] + verb_preds.values.squeeze().tolist()
    nouns = [noun_gt] + noun_t5
    noun_scores = [noun_scores] + noun_preds.values.squeeze().tolist()

    if "weights" in out.keys():
        weights = out["weights"].cpu().numpy()
    else:
        # in case of model without attention use dummy weights
        weights = np.zeros((cfg.test.num_segments, 1, 25))
    spec = spec.numpy().squeeze()
    fig, axarr = plt.subplots(4, cfg.test.num_segments, figsize=(16, 16))
    axarr[0, 0].set_ylabel("RGB Frames", fontsize=10)
    axarr[1, 0].set_ylabel("Audio Spectrograms", fontsize=10)
    axarr[2, 0].set_ylabel("Attention Weights", fontsize=10)
    axarr[3, 0].set_ylabel("Classes")
    for idx in range(cfg.test.num_segments):
        x = np.arange(weights.shape[2])
        img = Image.open(
            os.path.join(
                cfg.data_dir,
                cfg.data.rgb.dir_prefix,
                data["vid_id"][0],
                "img_{:010d}.jpg".format(rgb_indices[idx]),
            )
        )
        img = img.resize((256, 256))
        axarr[0, idx].imshow(img)
        tm = str(datetime.timedelta(seconds=rgb_indices[idx] / cfg.data.vid_fps))[0:-3]
        axarr[0, idx].set_title(f"Time: {tm}")
        axarr[1, idx].imshow(spec[idx], cmap="jet", origin="lowest", aspect="auto")
        axarr[2, idx].plot(x, weights[idx].squeeze(0))
        axarr[2, idx].set_ylim([0, 1])
        axarr[2, idx].set_xlim([0, weights.shape[2] - 1])

    axarr[3, 0].bar(
        np.arange(len(verbs)),
        verb_scores,
        width=0.5,
        align="center",
        alpha=1.0,
        color=["red", "blue", "green", "green", "green", "green"],
    )
    axarr[3, 0].set_xticks(np.arange(len(verbs)))
    axarr[3, 0].set_xticklabels(verbs)
    axarr[3, 0].set_title("Verbs")
    axarr[3, 1].bar(
        np.arange(len(nouns)),
        noun_scores,
        width=0.5,
        align="center",
        alpha=1.0,
        color=["red", "blue", "green", "green", "green", "green"],
    )
    axarr[3, 1].set_xticks(np.arange(len(nouns)))
    axarr[3, 1].set_xticklabels(nouns)
    axarr[3, 1].set_title("Nouns")

    axarr[3, 2].axis("off")

    fig.suptitle(f"Video Id: {data['vid_id'][0]}", fontsize=20)

    fig.savefig("results/vis.png")

    save_action_segment(
        cfg.data_dir, data["vid_id"][0], data["start_time"][0], data["stop_time"][0]
    )


def create_dataset(cfg, action_list=None):
    modality = get_modality(cfg)

    transforms = {}
    for m in modality:
        if m == "RGB":
            transforms[m] = torchvision.transforms.Compose(
                [
                    Rescale(cfg.data.test_scale_size),
                    CenterCrop(cfg.data.test_crop_size),
                    Stack(m),
                    ToTensor(),
                    Normalize(cfg.data.rgb.mean, cfg.data.rgb.std),
                ]
            )
        elif m == "Flow":
            transforms[m] = torchvision.transforms.Compose(
                [
                    Rescale(cfg.data.test_scale_size),
                    CenterCrop(cfg.data.test_crop_size),
                    Stack(m),
                    ToTensor(),
                    Normalize(cfg.data.flow.mean, cfg.data.flow.std),
                ]
            )
        elif m == "Audio":
            transforms[m] = torchvision.transforms.Compose(
                [Stack(m), ToTensor(is_audio=True)]
            )

    if cfg.test.vid_list:
        print("Reading list of test videos...")
        with open(os.path.join("./", cfg.test.vid_list)) as f:
            test_list = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        print("Done.")
        print("----------------------------------------------------------")
    else:
        test_list = None

    print("Creating the dataset using {}...".format(cfg.test.annotation_file[0]))
    dataset = Video_Dataset(
        cfg,
        test_list,
        cfg.test.annotation_file[0],
        modality,
        transform=transforms,
        mode="test",
        action_list=action_list,
    )
    print("Done.")
    print("----------------------------------------------------------")

    if len(dataset) > 0:
        return dataset
    else:
        print("!!!! No data found for these videos and actions. Please try again !!!!")
        return None


def initialize(config_file):
    """
    Initialize model , data loaders, loss function, optimizer and evaluate the model

    Args
    ----------
    cfg: dict
        Dictionary of config parameters
    modality: list
        List of input modalities

    """

    cfg = OmegaConf.load(config_file)

    np.random.seed(cfg.data.manual_seed)
    torch.manual_seed(cfg.data.manual_seed)

    modality = get_modality(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Initializing model...")
    model, _, num_gpus = build_model(cfg, modality, device)
    print("Model initialized.")
    print("----------------------------------------------------------")

    if cfg.test.pre_trained:
        if os.path.exists(cfg.test.pre_trained):
            print("Loading pre-trained weights {}...".format(cfg.test.pre_trained))
            data_dict = torch.load(cfg.test.pre_trained, map_location="cpu")
            if num_gpus > 1:
                model.module.load_state_dict(data_dict["model"])
            else:
                model.load_state_dict(data_dict["model"])
            print("Done.")
            print("----------------------------------------------------------")
        else:
            raise Exception(f"{cfg.test.pre_trained} file not found.")

    epic_classes = EpicClasses(os.path.join(cfg.data_dir, "annotations"))

    return cfg, model, epic_classes, device
