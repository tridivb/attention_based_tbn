import torch

from models.model import TBNModel

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
_LOSS_TYPES = {"CrossEntropy": torch.nn.CrossEntropyLoss, "NLL": torch.nn.NLLLoss}


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
        cfg.MODEL.ARCH in _MODEL_TYPES.keys()
    ), "Model type '{}' not supported".format(cfg.MODEL.ARCH)
    assert (
        cfg.MODEL.LOSS_FN in _LOSS_TYPES.keys()
    ), "Loss type '{}' not supported".format(cfg.MODEL.LOSS_FN)
    assert (
        cfg.NUM_GPUS <= torch.cuda.device_count()
    ), "Cannot use more GPU devices than available"

    # Construct the model
    model = _MODEL_TYPES[cfg.MODEL.ARCH](cfg, modality)

    # Set loss type
    criterion = _LOSS_TYPES[cfg.MODEL.LOSS_FN]()

    # Use multi-gpus if set in config
    if cfg.NUM_GPUS > 1 and device.type == "cuda":
        device_ids = cfg.GPU_IDS if len(cfg.GPU_IDS) > 0 else None
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        # criterion = torch.nn.DataParallel(criterion, device_ids=device_ids)

    model, criterion = model.to(device), criterion.to(device)

    return model, criterion
