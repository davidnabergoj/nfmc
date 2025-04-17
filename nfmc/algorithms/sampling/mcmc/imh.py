from typing import Union, Tuple, Optional, Dict, Any
from dataclasses import dataclass

import torch

from nfmc.algorithms.sampling.base import MCMCKernel
from nfmc.algorithms.sampling.mcmc.base import MetropolisSampler, MetropolisParameters, MetropolisKernel, MCMCSampler
from nfmc.util import sum_except_batch, metropolis_acceptance_log_ratio


@dataclass
class GaussianIMHKernel(MCMCKernel):
    """
    :param torch.Tensor proposal_mean_flat: mean of the Gaussian proposal with shape `(n_dim,)`.
    :param torch.Tensor proposal_sqrt_cov_flat: square root of the covariance of the Gaussian proposal with shape
     `(n_dim, n_dim)`.

    TODO make this a subclass of MetropolisKernel after splitting up MetropolisKernel into sensible subclasses.
    """
    proposal_mean_flat: Union[torch.Tensor, float] = 0.0
    proposal_sqrt_cov_flat: Union[torch.Tensor, float] = 1.0

    def __post_init__(self):
        if isinstance(self.proposal_sqrt_cov_flat, float):
            if self.proposal_sqrt_cov_flat <= 0:
                raise ValueError("proposal_sqrt_cov_flat must be positive if provided as float")

    def propose(self, x: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, float]]:
        """

        :param x:
        :return: x_prime and proposal potential.
        """
        batch_shape = x.shape[:-len(self.event_shape)]
        noise_flat = torch.randn(size=(*batch_shape, self.event_size,))
        sigma = self.proposal_sqrt_cov_flat
        mu = self.proposal_mean_flat

        if isinstance(sigma, float):
            multiplied_noise = noise_flat * sigma
        else:
            multiplied_noise = torch.einsum("ij,...j->...i", sigma, noise_flat)

        x_prime_flat = multiplied_noise + mu
        x_prime = x_prime_flat.view_as(x)
        return x_prime, 0.0


@dataclass
class GaussianIMHParameters(MetropolisParameters):
    pass


class GaussianIMH(MCMCSampler):
    """
    IMH with a Gaussian proposal distribution.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 target: callable,
                 kernel: Optional[GaussianIMHKernel] = None,
                 params: Optional[GaussianIMHParameters] = None):
        if kernel is None:
            n_dim = int(torch.prod(torch.as_tensor(event_shape)))
            kernel = GaussianIMHKernel(
                event_shape=event_shape,
                proposal_mean_flat=torch.zeros(size=(n_dim,)),
                proposal_sqrt_cov_flat=torch.eye(n_dim),
            )
        if params is None:
            params = GaussianIMHParameters()
        super().__init__(event_shape, target, kernel, params)
        self.u_x = None

    def propose(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
        """
        :param torch.Tensor x: current state with shape (*batch_shape, *event_shape).
        """
        self.kernel: GaussianIMHKernel

        batch_shape = x.shape[:-len(self.event_shape)]

        x_prime, _ = self.kernel.propose(x)

        # Divergence occurs if an element of x_prime is not finite
        divergence_mask = sum_except_batch((~torch.isfinite(x_prime)).long(), self.event_shape) > 0
        acceptance_mask = torch.zeros_like(divergence_mask)

        if self.params.adjustment:
            log_prob_accept = metropolis_acceptance_log_ratio(
                -self.target(x[~divergence_mask]),
                -self.target(x_prime[~divergence_mask]),
                0.0,
                0.0,
            )
            log_u = torch.randn_like(log_prob_accept).log()
            acceptance_mask[~divergence_mask] = log_u < log_prob_accept
        else:
            acceptance_mask[~divergence_mask] = True

        x_prime = x

        n_divergences = int(divergence_mask.long().sum())
        n_grads = 0
        n_calls = 0
        if self.params.adjustment:
            n_calls = 2 * torch.prod(torch.as_tensor(batch_shape))

        return x_prime.detach(), acceptance_mask, n_calls, n_grads, n_divergences

    def update_kernel(self, data: Dict[str, Any]):
        raise NotImplementedError
