import torch.nn as nn


class ContrastLoss(nn.Module):
    def __init__(self, reduction=None):
        super(ContrastLoss, self).__init__()
        if reduction in ["mean", "sum"]:
            self.reduction = reduction
        else:
            raise Exception(f"{reduction} type reduction not supported for Contrast Loss")

    def forward(self, input):
        binary_mask = input.detach()
        binary_mask[binary_mask >= 0.1] = 1
        binary_mask[binary_mask < 0.1] = 0
        
        loss = ((input * (1 - binary_mask)) - (input * binary_mask)).sum(dim=1)
        
        if self.reduction == "mean":
            loss = loss.mean()
        
        return loss
