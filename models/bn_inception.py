import torch
import torch.nn as nn
import torch.nn.functional as F
import os

import pretrainedmodels as ptm
from pretrainedmodels.models.bninception import BNInception


class BNInception(BNInception):
    def logits(self, features):
        adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        # x = self.last_linear(x)
        return x


def bninception(in_channels, modality, pretrained="imagenet", model_dir=""):
    """
        BNInception model architecture from <https://arxiv.org/pdf/1502.03167.pdf>`_ paper.
    """
    num_classes = 1000
    if pretrained is not None:
        if pretrained == "kinetics":
            num_classes = 400
            file = os.path.join(model_dir, "kinetics_bninception_flow.pth")

        elif pretrained == "imagenet":
            file = os.path.join(model_dir, "imagenet_bninception_rgb.pth")

        data_dict = torch.load(file, map_location="cpu")

    model = BNInception(num_classes=num_classes)
    model.feature_size = 1024

    if modality != "RGB":
        model.conv1_7x7_s2 = nn.Conv2d(
            in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)
        )
        if modality == "Audio":
            data_dict["conv1_7x7_s2.weight"] = (
                data_dict["conv1_7x7_s2.weight"].mean(dim=1).unsqueeze(dim=1)
            )

    model.load_state_dict(data_dict)

    return model
