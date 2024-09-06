import math
from dataclasses import dataclass

import torch


@dataclass
class DualAveragingParams:
    target_acceptance_rate: float = 0.651
    kappa: float = 0.75
    gamma: float = 0.05
    t0: int = 10


class DualAveraging:
    def __init__(self, initial_step_size, params: DualAveragingParams):
        self.t = params.t0
        self.error_sum = 0.0

        self.log_step_averaged = math.log(initial_step_size)
        self.log_step = math.inf
        self.mu = math.log(10 * initial_step_size)
        self.p = params

    def step(self, acceptance_rate_error):
        self.error_sum += float(acceptance_rate_error)  # This will eventually converge to 0 if all is well

        # Update raw step
        self.log_step = self.mu - self.error_sum / (math.sqrt(self.t) * self.p.gamma)

        # Update smoothed step
        eta = self.t ** -self.p.kappa
        self.log_step_averaged = eta * self.log_step + (1 - eta) * self.log_step_averaged
        self.t += 1

    @property
    def value(self):
        return math.exp(self.log_step_averaged)

    def __repr__(self):
        return f'DA error: {self.error_sum:.2f}'


def train_val_split(x: torch.Tensor,
                    train_pct: float,
                    max_train_size: int,
                    max_val_size: int,
                    shuffle: bool = True):
    """

    :param x: data with shape `(n_iterations, n_chains, *event_shape)`.
    :param train_pct:
    :param max_train_size:
    :param max_val_size:
    :param shuffle:
    :return:
    """
    x_train = x.flatten(0, 1)
    if shuffle:
        x_train = x_train[torch.randperm(len(x_train))]
    n_train = int(train_pct * len(x_train))
    x_train, x_val = x_train[:n_train], x_train[n_train:]
    x_train = x_train[:max_train_size]
    x_val = x_val[:max_val_size]
    return x_train, x_val
