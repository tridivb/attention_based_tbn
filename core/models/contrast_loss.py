import torch.nn as nn


class ContrastLoss(nn.Module):
    def __init__(self, threshold=0.5, reduction=None):
        super(ContrastLoss, self).__init__()
        self.threshold = threshold
        if reduction in ["mean", "batchmean", "sum"]:
            self.reduction = reduction
        else:
            raise Exception(
                f"{reduction} type reduction not supported for Contrast Loss"
            )

    def forward(self, input):
        binary_mask = input.clone().detach()
        binary_mask[binary_mask >= self.threshold] = 1
        binary_mask[binary_mask < self.threshold] = 0

        loss = ((input * (1 - binary_mask)) - (input * binary_mask)).sum(dim=1)

        if self.reduction in ["mean", "batchmean"]:
            loss = loss.mean()

        return loss
