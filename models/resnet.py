import torch
import torch.nn as nn
from torchvision import models
import numpy as np


class Resnet(nn.Module):
    def __init__(self, model_name, modality, in_channels):
        super(Resnet, self).__init__()

        if model_name == "resnet18":
            self.model = models.resnet18(pretrained=True)
        elif model_name == "resnet34":
            self.model = models.resnet34(pretrained=True)
        elif model_name == "resnet50":
            self.model = models.resnet50(pretrained=True)
        elif model_name == "resnet101":
            self.model = models.resnet101(pretrained=True)
        elif model_name == "resnet152":
            self.model = models.resnet152(pretrained=True)

        if modality != "RGB":
            self.model.conv1 = nn.Conv2d(
                in_channels,
                self.model.conv1.out_channels,
                kernel_size=self.model.conv1.kernel_size,
                stride=self.model.conv1.stride,
                padding=self.model.conv1.padding,
                bias=self.model.conv1.bias,
            )

        self.feature_size = self.model.fc.in_features
        self.model = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, input):
        feat = self.model(input)
        return feat
