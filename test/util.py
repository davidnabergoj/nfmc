import torch
from nfmc.util import sum_except_batch


class StandardGaussian:
    """
    Standard Gaussian distribution class.
    """

    def __init__(self, event_shape):
        self.event_shape = event_shape

    def neg_log_prob(self, x: torch.Tensor):
        """
        Computes the negative log probability density of this distribution.

        :param torch.Tensor x: input tensor with shape `(*batch_shape, *event_shape)`.
        :return: negative log probability density tensor with shape `batch_shape`.
        """
        return sum_except_batch(x**2, self.event_shape)


class DiagonalGaussian:
    """
    Class for the diagonal Gaussian distribution.
    """

    def __init__(self, event_shape, mu: float = 3.0, std: float = 2.0):
        self.event_shape = event_shape
        self.mu = mu
        self.std = std

    @property
    def first_moment(self):
        return torch.full(size=self.event_shape, fill_value=self.mu)

    @property
    def variance(self):
        return torch.full(size=self.event_shape, fill_value=self.std**2)

    @property
    def second_moment(self):
        return self.variance + self.first_moment**2

    def neg_log_prob(self, x: torch.Tensor):
        """
        Computes the negative log probability density of this distribution.

        :param torch.Tensor x: input tensor with shape `(*batch_shape, *event_shape)`.
        :return: negative log probability density tensor with shape `batch_shape`.
        """
        return sum_except_batch(
            (x - self.mu) ** 2 / (2 * self.std**2), self.event_shape
        )
