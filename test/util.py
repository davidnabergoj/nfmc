import torch


def standard_gaussian_potential(x):
    return torch.sum(x ** 2, dim=-1)


def diagonal_gaussian_potential(x):
    return torch.sum(x ** 2 / (2 * 100 ** 2), dim=-1)
