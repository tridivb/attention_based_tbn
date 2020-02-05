import os
import torch
import torch.nn as nn
import torchvision
import numpy as np

from .vgg import VGG
from .resnet import Resnet
from .bn_inception import bninception
from .consensus import SegmentConsensus


class TBNModel(nn.Module):
    """
    Temporal Binding Model

    Args
    ----------
    cfg: dict
        Dictionary of config parameters
    modality: list
        List of input modalities
    """

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

        # Create base model for each modality
        in_features = 0
        for m in self.modality:
            self.add_module("Base_{}".format(m), self._create_base_model(m))
            in_features += getattr(self, "Base_{}".format(m)).feature_size
            if cfg.MODEL.FREEZE_BASE:
                self._freeze_base_model(m, freeze_mode=cfg.MODEL.FREEZE_MODE)

        # Create fusion layer (if applicable) and final linear classificatin layer
        if len(self.modality) > 1:
            self.add_module(
                "fusion", Fusion(in_features, 512, dropout=cfg.MODEL.FUSION_DROPOUT)
            )
            self.add_module(
                "classifier", Classifier(self.num_classes, 512),
            )
        else:
            self.add_module(
                "classifier", Classifier(self.num_classes, in_features),
            )
        
        self.consensus = SegmentConsensus.apply

    def _create_base_model(self, modality):
        """
        Helper function to initialize the base model
        Args
        ----------
        modality: list
            List of input modalities

        Returns
        ----------
        base_model: torch.nn.model
            The base model
        """

        if modality == "RGB":
            in_channels = 3
        elif modality == "Flow":
            in_channels = 10
        elif modality == "Audio":
            in_channels = 1

        model_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(model_dir, "weights")

        if "vgg" in self.base_model_name:
            base_model = VGG(self.base_model_name, modality, in_channels)
        elif "resnet" in self.base_model_name:
            base_model = Resnet(self.base_model_name, modality, in_channels)
        elif self.base_model_name == "bninception":
            pretrained = "kinetics" if modality == "Flow" else "imagenet"
            base_model = bninception(
                in_channels,
                modality,
                model_dir=model_dir,
                pretrained=pretrained,
            )

        return base_model

    def _freeze_base_model(self, modality, freeze_mode):
        """
        Helper function to freeze weights of the base model
        Args
        ----------
        modality: list
            List of input modalities
        freeze_mode: str
            Mode of freezing

        """

        if freeze_mode == "all":
            print("Freezing the Base model.")
            for param in getattr(self, "Base_{}".format(modality)).parameters():
                param.requires_grad = False
        elif freeze_mode == "partialbn":
            print(
                "Freezing the batchnorms of Base Model {} except first layer.".format(
                    modality
                )
            )
            for mod_no, mod in enumerate(
                getattr(self, "Base_{}".format(modality)).children()
            ):
                if isinstance(mod, torch.nn.BatchNorm2d) and mod_no > 1:
                    mod.weight.requires_grad = False
                    mod.bias.requires_grad = False

    # def _aggregate_scores(self, scores, new_shape=(1, -1)):
    #     """
    #     Helper function to freeze weights of the base model
        
    #     Args
    #     ----------
    #     scores: tensor, dict
    #         Final output scores for each temporal binding window
    #     new_shape: tuple
    #         New shape for the output tensor

    #     """

    #     assert isinstance(scores, (dict, torch.Tensor))
    #     assert isinstance(new_shape, tuple)

    #     if isinstance(scores, dict):
    #         for key in scores.keys():
    #             # Reshape the tensor to B x N x feature size,
    #             # before calculating the mean over the trimmed action segment
    #             # where B = batch size and N = number of segment
    #             scores[key] = scores[key].view(new_shape).mean(dim=1)
    #     else:
    #         scores = scores[key].view(new_shape).mean(dim=1)

    #     return scores

    def forward(self, input):
        """
        Forward pass
        """
        features = []
        for m in self.modality:
            b, n, c, h, w = input[m].shape
            base_model = getattr(self, "Base_{}".format(m))
            feature = base_model(input[m].view(b * n, c, h, w))
            features.extend([feature.view(b * n, -1)])
        features = torch.cat(features, dim=1)

        if len(self.modality) > 1:
            features = self.fusion(features)

        out = self.classifier(features)

        for key in out.keys():
            out[key] = out[key].view((b, n, -1))
            out[key] = self.consensus(out[key]).squeeze()

        # out = self._aggregate_scores(out, new_shape=(b, n, -1))

        return out

    def get_loss(self, criterion, target, preds):
        """
        Helper function calculate loss for each classficiation layer
        and then sum them up

        Args
        ----------
        criterion: torch.nn.loss
            Loss function to use
        target: dict
            Dictionary of target values for each class type
        preds: dict
            Dictionary of predicted values for each class type

        Returns:
        ----------
        loss: dict
            Dictionary of losses for each class
        batch_size: int
            Current batch size

        """
        assert isinstance(target, dict)
        assert isinstance(preds, dict)

        loss = {"total": 0}

        for key in target.keys():
            labels = target[key]
            batch_size = target[key].shape[0]
            loss[key] = criterion(preds[key], labels)
            loss["total"] += loss[key]

        return loss, batch_size


class Fusion(nn.Module):
    """
    Fusion layer module
    """

    def __init__(self, in_size, out_size, dropout=0):
        super(Fusion, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.dropout = dropout

        self.fusion_layer = nn.Sequential(nn.Linear(in_size, out_size), nn.ReLU())
        torch.nn.init.normal_(self.fusion_layer[0].weight, 0, 1e-3)
        torch.nn.init.constant_(self.fusion_layer[0].bias, 0)

        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(p=self.dropout)

    def forward(self, input):
        out = self.fusion_layer(input)

        if self.dropout > 0:
            out = self.dropout_layer(out)

        return out


class Classifier(nn.Module):
    """
    Classifier layer module
    """

    def __init__(self, num_classes, in_features):
        super(Classifier, self).__init__()

        self.num_classes = num_classes

        for cls in num_classes.keys():
            self.add_module(cls, nn.Linear(in_features, self.num_classes[cls]))
            torch.nn.init.normal_(getattr(self, cls).weight, 0, 1e-3)
            torch.nn.init.constant_(getattr(self, cls).bias, 0)

    def forward(self, input):
        out = {}
        for cls in self.num_classes:
            classifier = getattr(self, cls)
            out[cls] = classifier(input)

        return out
