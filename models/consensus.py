import torch

# Inspired from https://github.com/yjxiong/tsn-pytorch/blob/master/ops/basic_ops.py
class SegmentConsensus(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        ctx.save_for_backward(input_tensor)
        return input_tensor.mean(dim=1, keepdim=True)
    
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_in = grad_output.expand(input.shape) / float(input.shape[1])
        return grad_in