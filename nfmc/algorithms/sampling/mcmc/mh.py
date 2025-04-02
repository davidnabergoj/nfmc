import math
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import torch

from nfmc.algorithms.sampling.mcmc.base import MetropolisKernel, MetropolisParameters, MetropolisSampler
from nfmc.util import metropolis_acceptance_log_ratio
from torchflows.utils import sum_except_batch


@dataclass
class MHKernel(MetropolisKernel):
    event_size: int

    def __repr__(self):
        return (f'log step: {math.log(self.step_size):.2f}, '
                f'mass norm: {torch.max(torch.abs(self.inv_mass_diag)):.2f}')


@dataclass
class MHParameters(MetropolisParameters):
    imd_adjustment: float = 1e-5

    def __post_init__(self):
        self.tune_step_size = False
        self.tune_inv_mass_diag = True


class MH(MetropolisSampler):
    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 target: callable,
                 kernel: Optional[MHKernel] = None,
                 params: Optional[MHParameters] = None):
        if kernel is None:
            kernel = MHKernel(event_size=int(torch.prod(torch.as_tensor(event_shape))))
        if params is None:
            params = MHParameters()
        super().__init__(event_shape, target, kernel, params)

    @property
    def name(self):
        return "MH"

    def propose(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
        """
        :param torch.Tensor x: current state with shape (*batch_shape, *event_shape).
        """
        batch_shape = x.shape[:-len(self.event_shape)]

        noise = torch.multiply(
            torch.randn(size=(*batch_shape, self.event_size)),
            self.kernel.inv_mass_diag[[None] * len(batch_shape)]
        ).view_as(x)
        x_prime = x + self.kernel.step_size * noise

        # Divergence occurs if an element of x_prime is not finite
        divergence_mask = sum_except_batch((~torch.isfinite(x_prime)).long(), self.event_shape) > 0
        acceptance_mask = torch.zeros_like(divergence_mask)

        if self.params.adjustment:
            log_prob_accept = metropolis_acceptance_log_ratio(
                -self.target(x[~divergence_mask]),
                -self.target(x_prime[~divergence_mask]),
                0,
                0
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


class RandomWalk(MH):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params.adjustment = False
