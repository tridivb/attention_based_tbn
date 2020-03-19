import torch
import torch.nn as nn
from torchvision import models
import numpy as np


class Resnet(nn.Module):
    """
    Resnet module
    """

    def __init__(self, model_depth, modality, in_channels):
        super(Resnet, self).__init__()

        if model_depth == 18:
            self.model = models.resnet18(pretrained=True)
        elif model_depth == 34:
            self.model = models.resnet34(pretrained=True)
        elif model_depth == 50:
            self.model = models.resnet50(pretrained=True)
        elif model_depth == 101:
            self.model = models.resnet101(pretrained=True)
        elif model_depth == 152:
            self.model = models.resnet152(pretrained=True)

        if modality != "RGB":
            weight = self.model.conv1.weight.mean(dim=1).unsqueeze(dim=1)
            self.model.conv1 = nn.Conv2d(
                in_channels,
                self.model.conv1.out_channels,
                kernel_size=self.model.conv1.kernel_size,
                stride=self.model.conv1.stride,
                padding=self.model.conv1.padding,
                bias=self.model.conv1.bias,
            )
            self.model.conv1.weight = torch.nn.Parameter(weight)

        self.feature_size = self.model.fc.in_features
        self.model = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, input):
        feat = self.model(input)
        return feat
