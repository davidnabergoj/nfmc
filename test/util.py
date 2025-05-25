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
        return sum_except_batch(
            x ** 2,
            self.event_shape
        )


class DiagonalGaussian:
    """
    Class for the diagonal Gaussian distribution with mean zero standard deviation equal 100 in all dimensions.
    """

    def __init__(self, event_shape):
        self.event_shape = event_shape

    def neg_log_prob(self, x: torch.Tensor):
        """
        Computes the negative log probability density of this distribution.

        :param torch.Tensor x: input tensor with shape `(*batch_shape, *event_shape)`.
        :return: negative log probability density tensor with shape `batch_shape`.
        """
        return sum_except_batch(
            x ** 2 / (2 * 100 ** 2),
            self.event_shape
        )
