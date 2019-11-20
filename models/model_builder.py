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
    "bninception": TBNModel,
}


def build_model(cfg):
    """
    Builds the model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone.
    """
    assert (
        cfg.MODEL.ARCH in _MODEL_TYPES.keys()
    ), "Model type '{}' not supported".format(cfg.MODEL.ARCH)
    assert (
        cfg.NUM_GPUS <= torch.cuda.device_count()
    ), "Cannot use more GPU devices than available"

    # Construct the model
    model = _MODEL_TYPES[cfg.MODEL.ARCH](cfg)
    # Determine the GPU used by the current process
    cur_device = torch.cuda.current_device()
    # Transfer the model to the current GPU device
    model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device
        )
    return model
