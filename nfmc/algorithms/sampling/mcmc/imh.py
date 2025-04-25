import math

from typing import Union, Tuple, Optional, Dict, Any
from dataclasses import dataclass

import torch

from nfmc.algorithms.sampling.base import MCMCKernel
from nfmc.algorithms.sampling.mcmc.base import MetropolisParameters, MCMCSampler
from nfmc.algorithms.sampling.tuning import DualAveraging, DualAveragingParams
from nfmc.util import sum_except_batch, metropolis_acceptance_log_ratio
from potentials.synthetic.gaussian.diagonal import gaussian_potential


@dataclass
class GaussianIMHParameters(MetropolisParameters):
    pass


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

    def __repr__(self):
        return ""

    def proposal_potential(self, x: torch.Tensor) -> Union[torch.Tensor, float]:
        sqrt_cov = self.proposal_sqrt_cov_flat
        mu = self.proposal_mean_flat

        batch_shape = x.shape[:-len(self.event_shape)]

        x_flat = x.view(*batch_shape, -1)

        cov = torch.einsum('ij,jk->ik', sqrt_cov.T, sqrt_cov)
        cov_inv = torch.linalg.inv(cov)
        return -0.5 * torch.logdet(cov) - 0.5 * torch.einsum('...i,ij,...j->...', x_flat - mu, cov_inv, x_flat - mu)

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

        u_x_prime = self.proposal_potential(x_prime)
        return x_prime, u_x_prime

    def update(self, data: Dict[str, Any]):
        raise NotImplementedError


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
                target=target,
                proposal_mean_flat=torch.zeros(size=(n_dim,)),
                proposal_sqrt_cov_flat=torch.eye(n_dim),
            )
        if params is None:
            params = GaussianIMHParameters()
        super().__init__(event_shape, target, kernel, params)

    def propose(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
        """
        :param torch.Tensor x: current state with shape (*batch_shape, *event_shape).
        """
        self.kernel: GaussianIMHKernel

        batch_shape = x.shape[:-len(self.event_shape)]

        x_prime, u_x_prime = self.kernel.propose(x)
        u_x = self.kernel.proposal_potential(x)

        # Divergence occurs if an element of x_prime is not finite
        divergence_mask = sum_except_batch((~torch.isfinite(x_prime)).long(), self.event_shape) > 0
        acceptance_mask = torch.zeros_like(divergence_mask)

        if self.params.adjustment:
            log_prob_accept = metropolis_acceptance_log_ratio(
                -self.target(x[~divergence_mask]),
                -self.target(x_prime[~divergence_mask]),
                -u_x[~divergence_mask],
                -u_x_prime[~divergence_mask],
            )
            log_u = torch.rand_like(log_prob_accept).log()
            acceptance_mask[~divergence_mask] = log_u < log_prob_accept
        else:
            acceptance_mask[~divergence_mask] = True

        n_divergences = int(divergence_mask.long().sum())
        n_grads = 0
        n_calls = 0
        if self.params.adjustment:
            n_calls = 2 * torch.prod(torch.as_tensor(batch_shape))

        return x_prime.detach(), acceptance_mask, n_calls, n_grads, n_divergences


@dataclass
class IsotropicGaussianIMHParameters(GaussianIMHParameters):
    tune_proposal_scale: bool = False

    def __post_init__(self):
        if self.tune_proposal_scale:
            raise ValueError


@dataclass
class IsotropicGaussianIMHKernel(GaussianIMHKernel):
    proposal_sqrt_cov_flat: float = 1.0

    def __repr__(self):
        return f"Proposal log scale: {math.log(self.proposal_sqrt_cov_flat):.3f}"

    def proposal_potential(self, x: torch.Tensor) -> Union[torch.Tensor, float]:
        sigma = self.proposal_sqrt_cov_flat
        mu = self.proposal_mean_flat

        batch_shape = x.shape[:-len(self.event_shape)]

        x_flat = x.view(*batch_shape, -1)
        return gaussian_potential(x_flat, torch.as_tensor(mu), torch.as_tensor(sigma)).sum(dim=-1)
    
    def __post_init__(self):
        assert isinstance(self.proposal_sqrt_cov_flat, float), f"{type(self.proposal_sqrt_cov_flat) = }"

class IsotropicGaussianIMH(GaussianIMH):
    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 target: callable,
                 kernel: Optional[IsotropicGaussianIMHKernel] = None,
                 params: Optional[IsotropicGaussianIMHParameters] = None):
        if kernel is None:
            kernel = IsotropicGaussianIMHKernel(event_shape=event_shape, target=target)
        if params is None:
            params = IsotropicGaussianIMHParameters()
        super().__init__(event_shape, target, kernel, params)

    def update_kernel(self, data: Dict[str, Any]):
        raise ValueError


if __name__ == '__main__':
    torch.manual_seed(0)
    _event_shape = (2,)

    target_mu = 0.1
    target_std = 1.0
    _target_callable = lambda x: torch.sum((x - target_mu) ** 2 / (2 * target_std ** 2), dim=1)

    _params = IsotropicGaussianIMHParameters(
        n_iterations=1000,
        n_warmup_iterations=1000,
        tune_proposal_scale=True,
    )
    _kernel = IsotropicGaussianIMHKernel(
        _event_shape,
        _target_callable,
        proposal_sqrt_cov_flat=2.0
    )

    _sampler = IsotropicGaussianIMH(
        _event_shape,
        _target_callable,
        params=_params,
        kernel=_kernel,
    )
    _warmup_out = _sampler.warmup(x0=torch.randn(size=(10, *_event_shape)))
    _sampling_out = _sampler.sample(x0=torch.randn(size=(10, *_event_shape)))

    import matplotlib.pyplot as plt

    _x = _sampling_out.samples.flatten(0, 1)
    _x = _x[torch.randperm(len(_x))[:10000]]
    _x_true = torch.randn(size=(10000, 2))

    print(f'{_x.mean():.3f}')
    print(f'{_x_true.mean():.3f}')

    print(f'{_x.var():.3f}')
    print(f'{_x_true.var():.3f}')

    fig, ax = plt.subplots()
    ax.scatter(_x_true[:, 0], _x_true[:, 1], s=3, label='True')
    ax.scatter(_x[:, 0], _x[:, 1], s=3, label='IMH')
    ax.legend()
    plt.show()
