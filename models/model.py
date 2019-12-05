import os
import torch
import torch.nn as nn
import torchvision
import numpy as np

from models.vgg import VGG
from models.resnet import Resnet
from models.bn_inception import bninception


class TBNModel(nn.Module):
    def __init__(self, cfg, modality):
        super(TBNModel, self).__init__()

        self.cfg = cfg
        self.modality = modality
        self.base_model_name = cfg.MODEL.ARCH
        self.num_classes = cfg.MODEL.NUM_CLASSES
        if cfg.MODEL.AGG_TYPE.lower() == "avg":
            self.agg_type = "avg"
        else:
            print("Incorrect aggregation type")
            self.agg_type = None

        in_features = 0
        for m in self.modality:
            self.add_module("Base_{}".format(m), self._create_base_model(m))
            in_features += getattr(self, "Base_{}".format(m)).feature_size
            if cfg.MODEL.FREEZE_BASE:
                self._tune_base_model(modality, requires_grad=False)

        if len(self.modality) > 1:
            self.add_module(
                "fusion", Fusion(in_features, 512, dropout=cfg.MODEL.FUSION_DROPOUT)
            )
            self.add_module(
                "classifier",
                Classifier(self.num_classes, 512, use_softmax=cfg.MODEL.USE_SOFTMAX),
            )
        else:
            self.add_module(
                "classifier",
                Classifier(
                    self.num_classes, in_features, use_softmax=cfg.MODEL.USE_SOFTMAX
                ),
            )

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
            pretrained = "kinetics" if modality == "Flow" else "imagenet"
            base_model = bninception(
                in_channels,
                modality,
                model_dir=self.cfg.MODEL.CHECKPOINT_DIR,
                pretrained=pretrained,
            )

        return base_model

    def _tune_base_model(self, modality, requires_grad=False):
        for param in getattr(self, modality):
            param.requires_grad = requires_grad

    def _aggregate_scores(self, scores, new_shape=(1, -1)):
        assert isinstance(scores, (dict, torch.Tensor))
        assert isinstance(new_shape, tuple)

        if isinstance(scores, dict):
            for key in scores.keys():
                scores[key] = scores[key].view(new_shape).mean(dim=1)
        else:
            scores = scores[key].view(new_shape).mean(dim=1)

        return scores

    def forward(self, input):
        features = []
        for m in self.modality:
            b, n, c, h, w = input[m].shape
            base_model = getattr(self, "Base_{}".format(m))
            feature = base_model(input[m].view(b * n, c, h, w))
            features.extend([feature.view(b * n, -1)])
        features = torch.cat(features, dim=1)

        if self.fusion:
            features = self.fusion(features)

        out = self.classifier(features)

        out = self._aggregate_scores(out, new_shape=(b, n, -1))

        return out

    def get_loss(self, criterion, target, preds):
        assert isinstance(target, dict)
        assert isinstance(preds, dict)

        loss = 0

        for key in target.keys():
            labels = target[key]
            loss += criterion(preds[key], labels)

        return loss


class Fusion(nn.Module):
    def __init__(self, in_size, out_size, dropout=0):
        super(Fusion, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.dropout = dropout

        self.fusion_layer = nn.Sequential(nn.Linear(in_size, out_size), nn.ReLU())

        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(p=self.dropout)

    def forward(self, input):
        out = self.fusion_layer(input)

        if self.dropout > 0:
            out = self.dropout_layer(out)

        return out


class Classifier(nn.Module):
    def __init__(self, num_classes, in_features, use_softmax=False):
        super(Classifier, self).__init__()

        self.num_classes = num_classes
        self.use_softmax = use_softmax

        for cls in num_classes.keys():
            self.add_module(cls, nn.Linear(in_features, self.num_classes[cls]))
            if self.use_softmax:
                self.add_module("{}_softmax".format(cls), nn.Softmax(dim=1))

    def forward(self, input):
        out = {}
        for cls in self.num_classes:
            classifier = getattr(self, cls)
            out[cls] = classifier(input)
            if self.use_softmax:
                softmax = getattr(self, "{}_softmax".format(cls))
                out[cls] = softmax(out[cls])

        return out
