import torch


class GaussianNormalizer():
    def __init__(self, x: torch.Tensor):
        self.mean, self.std = x.mean(0), x.std(0)
        self.std[torch.where(self.std == 0.)] = 1.

    def normalize(self, x: torch.Tensor):
        return (x - self.mean[None,]) / self.std[None,]

    def unnormalize(self, x: torch.Tensor):
        return x * self.std[None,] + self.mean[None,]


class MaxMinNormlizer():
    def __init__(self, x: torch.Tensor):
        self.min = x.min()
        self.scale = x.max() - x.min()

    def normalize(self, x: torch.Tensor):
        return (x - self.min[None,]) / self.scale[None,]

    def unnormalize(self, x: torch.Tensor):
        return x * self.scale[None,] + self.min[None,]
