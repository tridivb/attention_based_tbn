import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from models.vgg import VGG
from models.resnet import Resnet
from models.bn_inception import BNInception


class TBNModel(nn.Module):
    def __init__(self, cfg, modality):
        super(Model, self).__init__()

        self.cfg = cfg
        self.modality = modality
        self.base_model_name = cfg.MODEL.ARCH
        self.num_classes = cfg.MODEL.NUM_CLASSES

        self.base_model = {}

        in_features = 0
        for m in self.modality:
            if m == "RGB":
                in_channels = 3
            elif m == "Flow":
                in_channels = 10
            elif m == "Audio":
                in_channels = 1
            self.base_model[m] = self._create_base_model(in_channels)
            in_features += self.base_model[m].feature_size

        if len(self.modality) > 1:
            in_features = 1024 * len(self.modality)
            self.fusion_layer = self._create_fusion_layer(in_features)
            self.classifier = self._create_classifier(512)
        else:
            self.classifier = self._create_classifier(1024)

    def _create_base_model(self, in_channels):
        if "vgg" in self.model:
            base_model = VGG(
                self.model,
                self.num_classes,
                self.mode,
                in_channels
            )
        elif "resnet" in self.model:
            base_model = Resnet(
                self.model,
                self.num_classes,
                self.mode,
                in_channels
            )
        elif self.model == "inception":
            base_model = BNInception(in_channels, self.num_classes)
        return base_model

    def _create_fusion_layer(self, in_features):
        return nn.Linear(in_features, 512)

    def _create_classifier(self, in_features):
        fc = {}
        if isinstance(self.num_classes, int):
            fc["classifier"] = nn.Linear(in_features, self.num_classes)
        elif isinstance(self.num_classes, list):
            for index in range(len(self.num_classes)):
                fc["classifer_{}".format(index)] = nn.Linear(in_features, self.num_classes[index])

        return fc

    def forward(self, input):
        for i, m in enumerate(self.modality):
            if i == 0:
                features = self.base_model(input).reshape(input.shape[0], -1)
            else:
                features = torch.cat((features, self.base_model(input).reshape(input.shape[0], -1)), dim=1)

        out = {}
        for key in self.classifier.keys():
            out[key] = self.classifier[key](features))
        return out