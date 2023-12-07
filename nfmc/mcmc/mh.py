from typing import Tuple

import torch

from nfmc.mcmc.base import Sampler
from nfmc.util import metropolis_acceptance_log_ratio


class MetropolisHastings(Sampler):
    def __init__(self,
                 n_dim: int,
                 potential: callable,
                 proposal_scale: torch.Tensor = None):
        super().__init__(n_dim, potential)
        if proposal_scale is None:
            proposal_scale = torch.ones([1, n_dim], dtype=torch.float)
        assert proposal_scale.shape == (1, n_dim)
        self.proposal_scale = proposal_scale

    def step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # metropolis hastings step with a diagonal normal proposal
        noise = torch.randn_like(x) * self.proposal_scale
        x_prime = x + noise
        log_alpha = metropolis_acceptance_log_ratio(
            -self.potential(x),
            -self.potential(x_prime),
            0,  # Can set to 0, will get cancelled out with log_proposal_prime
            0  # Can set to 0, will get cancelled out with log_proposal_curr
        )
        return x_prime, log_alpha


def mh(x0: torch.Tensor,
       potential: callable,
       full_output: bool = True,
       step_size: float = 0.01,
       **kwargs):
    n_dim = x0.shape[1]
    obj = MetropolisHastings(n_dim, potential, proposal_scale=torch.full(size=(1, n_dim), fill_value=step_size))
    return obj.sample(x0, full_output=full_output, **kwargs)
