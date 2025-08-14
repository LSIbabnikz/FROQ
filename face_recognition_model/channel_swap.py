
import torch

class PermuteChannels:
    def __init__(self, order):
        self.order = order

    def __call__(self, x):
        # x is a torch.Tensor, e.g., shape (C, H, W)
        return x[self.order, :, :]
    