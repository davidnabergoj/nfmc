import torch


def standard_gaussian_neg_log_prob(x):
    # x.shape == (batch_size, *event_shape)
    return torch.sum(x ** 2, dim=list(range(1, len(x.shape))))


def diagonal_gaussian_neg_log_prob(x):
    # x.shape == (batch_size, *event_shape)
    return torch.sum(x ** 2 / (2 * 100 ** 2), dim=list(range(1, len(x.shape))))
