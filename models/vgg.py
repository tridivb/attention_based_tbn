import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from torch.hub import load_state_dict_from_url


class VGG(nn.Module):
    def __init__(self, model_name, modality, in_channels):
        super(VGG, self).__init__()

        if model_name == "vgg16":
            self.model = models.vgg16(pretrained=True)
        elif model_name == "vgg16bn":
            self.model = models.vgg16_bn(pretrained=True)
        elif model_name == "vgg11":
            self.model = models.vgg11(pretrained=True)
        elif model_name == "vgg11bn":
            self.model = models.vgg11_bn(pretrained=True)

        if modality != "RGB":
            self.model.features[0] = nn.Conv2d(
                in_channels,
                self.model.features[0].out_channels,
                kernel_size=self.model.features[0].kernel_size,
                stride=self.model.features[0].stride,
                padding=self.model.features[0].padding,
            )

        self.feature_size = self.model.classifier[-1].in_features
        self.model.classifier = nn.Sequential(
            *list(self.model.classifier.children())[:-1]
        )

    def forward(self, input):
        feat = self.model(input)
        return feat
