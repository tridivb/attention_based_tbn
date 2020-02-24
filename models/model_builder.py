import torch

from .model import TBNModel
from .dataparallel import DataParallel

# Supported model types
_MODEL_TYPES = {
    "vgg11": TBNModel,
    "vgg11bn": TBNModel,
    "vgg16": TBNModel,
    "vgg16bn": TBNModel,
    "resnet18": TBNModel,
    "resnet34": TBNModel,
    "resnet50": TBNModel,
    "resnet101": TBNModel,
    "resnet152": TBNModel,
    "bninception": TBNModel,
}

# Supported loss types
_LOSS_TYPES = {"crossentropy": torch.nn.CrossEntropyLoss, "nll": torch.nn.NLLLoss}


def build_model(cfg, modality, device):
    """
    Helper function to build the model and initialize loss function. 
    All sanity checks for model parameters should be done here before initializing the model

    Args
    ----------
    cfg: dict
        Dictionary of config parameters
    modality: list
        List of input modalities
    device: torch.device
        Torch device to use
    """

    assert (
        cfg.model.arch in _MODEL_TYPES.keys()
    ), "Model type '{}' not supported".format(cfg.model.arch)
    assert (
        cfg.model.loss_fn in _LOSS_TYPES.keys()
    ), "Loss type '{}' not supported".format(cfg.model.loss_fn)
    if len(cfg.gpu_ids) > 0:
        num_gpus = len(cfg.gpu_ids)
    else:
        num_gpus = torch.cuda.device_count()
    assert (
        num_gpus <= torch.cuda.device_count()
    ), "Cannot use more GPU devices than available"

    # Construct the model
    model = _MODEL_TYPES[cfg.model.arch](cfg, modality, device)

    # Set loss type
    criterion = _LOSS_TYPES[cfg.model.loss_fn]()

    # Use multi-gpus if set in config
    if num_gpus > 1 and device.type == "cuda":
        device_ids = cfg.gpu_ids if len(cfg.gpu_ids) > 1 else None
        model = DataParallel(model, device_ids=device_ids)
        # criterion = torch.nn.DataParallel(criterion, device_ids=device_ids)

    model, criterion = model.to(device), criterion.to(device)

    return model, criterion, num_gpus
