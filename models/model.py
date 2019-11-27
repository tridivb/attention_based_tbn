import torch
import torch.nn as nn
import torchvision
import numpy as np

from models.vgg import VGG
from models.resnet import Resnet
from models.bn_inception import bninception
from utils.transform import MultiScaleCrop, RandomHorizontalFlip, ToTensor


class TBNModel(nn.Module):
    def __init__(self, cfg, modality):
        super(TBNModel, self).__init__()

        self.cfg = cfg
        self.modality = modality
        self.base_model_name = cfg.MODEL.ARCH
        self.num_classes = cfg.MODEL.NUM_CLASSES

        in_features = 0
        for m in self.modality:
            self.add_module("Base_{}".format(m), self._create_base_model(m))
            in_features += getattr(self, "Base_{}".format(m)).feature_size
            if cfg.MODEL.FREEZE_BASE:
                self._tune_base_model(modality, requires_grad=False)

        if len(self.modality) > 1:
            self.add_module("fusion_layer", self._create_fusion_layer(in_features))
            self.add_module("classifier", Classifier(self.num_classes, 512))
        else:
            self.add_module("classifier", Classifier(self.num_classes, in_features))

    def _create_base_model(self, modality):
        if modality == "RGB":
            in_channels = 3
        elif modality == "Flow":
            in_channels = 10
        elif modality == "Audio":
            in_channels = 1

        if "vgg" in self.base_model_name:
            base_model = VGG(self.base_model_name, modality, in_channels)
        elif "resnet" in self.base_model_name:
            base_model = Resnet(self.base_model_name, modality, in_channels)
        elif self.base_model_name == "bninception":
            base_model = bninception(in_channels, modality, pretrained="imagenet")

        return base_model

    def _create_fusion_layer(self, in_features):
        return nn.Linear(in_features, 512)

    def _tune_base_model(self, modality, requires_grad=False):
        for param in getattr(self, modality):
            param.requires_grad = requires_grad

    def forward(self, input, num_segments=3):
        for i in range(num_segments):
            features = []
            for m in self.modality:
                batch_size = input[m].shape[0]            
                base_model = getattr(self, "Base_{}".format(m))
                features.extend([base_model(input[m][:, i, :, :, :]).reshape(batch_size, -1)])
            features = torch.cat(features, dim=1)
            print(features.shape)

            if self.fusion_layer:
                features = self.fusion_layer(features)

            out = self.classifier(features)

        return out


class Classifier(nn.Module):
    def __init__(self, num_classes, in_features):
        super(Classifier, self).__init__()

        for class_name in num_classes.keys():
            self.add_module(class_name, nn.Linear(in_features, num_classes[class_name]))

        self.num_classes = num_classes

    def forward(self, input):
        out = {}
        for key in self.num_classes:
            classifier = getattr(self, key)
            out[key] = classifier(input)
        return out
