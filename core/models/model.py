import os
import torch
import torch.nn as nn
import torchvision
import numpy as np

from .vgg import VGG
from .resnet import Resnet
from .bn_inception import bninception
from .attention import PositionalEncoding, AttentionLayer


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

    def __init__(self, cfg, modality, device):
        super(TBNModel, self).__init__()

        self.cfg = cfg
        self.modality = modality
        self.base_model_name = cfg.model.arch
        self.num_classes = cfg.model.num_classes
        self.use_attention = cfg.model.attention.enable

        if cfg.model.agg_type.lower() == "avg":
            self.agg_type = "avg"
        else:
            print("Incorrect aggregation type")
            self.agg_type = None

        # Create base model for each modality
        in_features = 0
        for m in self.modality:
            self.add_module("Base_{}".format(m), self._create_base_model(m))
            in_features += getattr(self, "Base_{}".format(m)).feature_size
            if cfg.model.freeze_base:
                self._freeze_base_model(m, freeze_mode=cfg.model.freeze_mode)

        # Create fusion layer (if applicable) and final linear classificatin layer
        if len(self.modality) > 1:
            if self.use_attention:
                self.pe = nn.Sequential(
                    PositionalEncoding(10, max_len=25, device=device),
                    nn.Conv1d(1034, 1024, kernel_size=1),
                    # nn.BatchNorm1d(1024),
                    nn.GroupNorm(64, 1024),
                )
                self.attention_layer = AttentionLayer(
                    1024, cfg.model.attention.attn_heads, 0.5
                )
            self.add_module(
                "fusion", Fusion(in_features, 512, dropout=cfg.model.fusion_dropout)
            )
            self.add_module(
                "classifier", Classifier(self.num_classes, 512),
            )
        else:
            self.add_module(
                "classifier", Classifier(self.num_classes, in_features),
            )

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

        model_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        model_dir = os.path.join(model_dir, "weights")

        is_audio = True if modality == "Audio" else False

        if "vgg" in self.base_model_name:
            base_model = VGG(self.cfg.model.vgg.type, modality, in_channels)
        elif "resnet" in self.base_model_name:
            base_model = Resnet(cfg.model.resnet.depth, modality, in_channels)
        elif self.base_model_name == "bninception":
            pretrained = "kinetics" if modality == "Flow" else "imagenet"
            base_model = bninception(
                in_channels,
                modality,
                model_dir=model_dir,
                pretrained=pretrained,
                is_audio=is_audio,
                attend=self.use_attention,
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

    def _aggregate_scores(self, scores, new_shape=(1, -1)):
        """
        Helper function to freeze weights of the base model
        
        Args
        ----------
        scores: tensor, dict
            Final output scores for each temporal binding window
        new_shape: tuple
            New shape for the output tensor

        """

        assert isinstance(scores, (dict, torch.Tensor))
        assert isinstance(new_shape, tuple)

        if isinstance(scores, dict):
            for key in scores.keys():
                # Reshape the tensor to B x N x feature size,
                # before calculating the mean over the trimmed action segment
                # where B = batch size and N = number of segment
                scores[key] = scores[key].view(new_shape).mean(dim=1)
        else:
            scores = scores[key].view(new_shape).mean(dim=1)

        return scores

    def forward(self, input):
        """
        Forward pass
        """
        features = []
        for m in self.modality:
            b, n, c, h, w = input[m].shape
            base_model = getattr(self, "Base_{}".format(m))
            feature = base_model(input[m].view(b * n, c, h, w))
            if m == "Audio" and self.use_attention:
                feature = self.pe(feature)
                feature = feature.transpose(1, 2).transpose(0, 1)
                feature, att_wts = self.attention_layer(
                    features[0].unsqueeze(0), feature, feature
                )
                feature = feature.squeeze(0)
            features.extend([feature])
        features = torch.cat(features, dim=1)

        if len(self.modality) > 1:
            features = self.fusion(features)

        out = self.classifier(features)

        out = self._aggregate_scores(out, new_shape=(b, n, -1))

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