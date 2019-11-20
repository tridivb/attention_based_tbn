import torch
import torch.nn as nn
from torchvision import models
import numpy as np


class Resnet(nn.Module):
    def __init__(self, model, num_classes, mode, in_channels):
        super(Resnet, self).__init__()

        if model == "resnet18":
            self.model = models.resnet18(pretrained=True)
        elif model == "resnet34":
            self.model = models.resnet34(pretrained=True)
        elif model == "resnet50":
            self.model = models.resnet50(pretrained=True)
        elif model == "resnet101":
            self.model = models.resnet101(pretrained=True)

        if mode != "RGB":
            self.model.Conv2d_1a_3x3.conv = nn.Conv2d(
                in_channels,
                self.model.features[0].out_channels,
                kernel_size=self.model.features[0].kernel_size,
                stride=self.model.features[0].stride,
                padding=self.model.features[0].padding,
            )

        self.feature_size = self.model.fc.in_features
        self.model = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, input):
        feat = self.model(input)
        return feat
