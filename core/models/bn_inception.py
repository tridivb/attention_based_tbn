import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pretrainedmodels as ptm
from pretrainedmodels.models.bninception import BNInception

from .bn_inception_audio import BNInception_Audio


class BNInception(BNInception):
    """
    Inherited BNInception class from the pretrainedmodels module
    """

    def logits(self, features):
        """
        Overloaded logits function to return features from BNInception
        instead of class predictions
        """

        adaptiveAvgPoolWidth = features.shape[2:]
        if self.is_audio and self.attend:
            # Avg pool the spectrogram along frequency dimension only
            x = F.avg_pool2d(
                features,
                kernel_size=(adaptiveAvgPoolWidth[0], 1),
                stride=(adaptiveAvgPoolWidth[0], 1),
            )
            return x
        else:
            x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
            x = x.view(x.size(0), -1)
            # x = self.last_linear(x)
            return x


def bninception(
    in_channels,
    modality,
    pretrained="imagenet",
    model_dir="",
    is_audio=False,
    attend=False,
):
    """
    Initialize the BNInception model and cut off the final linear layer
    
    Args
    ----------
    in_channels: int
        Number of input channels
    modality: list
        List of input modalities
    pretrained: str, default = "imagenet"
        Pretrained model to initialize with
    model_dir: str, default = ""
        Location of pretrained model weights
    """
    num_classes = 1000
    if pretrained is not None:
        if pretrained == "kinetics":
            # For flow, use pretrained kinetics weights
            num_classes = 400
            file = os.path.join(model_dir, "kinetics_bninception_flow.pth")

        elif pretrained == "imagenet":
            file = os.path.join(model_dir, "imagenet_bninception_rgb.pth")

        data_dict = torch.load(file, map_location="cpu")

    if modality == "Audio":
        # model = BNInception_Audio(num_classes=num_classes, attend=attend)
        model = BNInception(num_classes=num_classes)
        model.conv1_7x7_s2 = nn.Conv2d(
                in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)
            )
        data_dict["conv1_7x7_s2.weight"] = (
            data_dict["conv1_7x7_s2.weight"].mean(dim=1).unsqueeze(dim=1)
        )
        missing_keys = [
            k for k in model.state_dict().keys() if k not in data_dict.keys()
        ]
        extra_keys = [k for k in data_dict.keys() if k not in model.state_dict().keys()]
        for key in missing_keys:
            data_dict[key] = model.state_dict()[key]
        for key in extra_keys:
            del data_dict[key]
    else:
        model = BNInception(num_classes=num_classes)
        if modality == "Flow":
            # Configure first convolution layer and its weights according to modality and input channels
            model.conv1_7x7_s2 = nn.Conv2d(
                in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)
            )

    model.is_audio = is_audio
    model.attend = attend
    model.feature_size = 1024

    model.load_state_dict(data_dict)
    print(f"Model initialized with {pretrained} weights")

    # Remove the linear classficiation layer
    delattr(model, "last_linear")

    return model
