from copy import deepcopy
from typing import Tuple

import torch


class Sampler:
    def __init__(self, n_dim: int, potential: callable, adjusted: bool = True):
        self.n_dim = n_dim
        self.potential = potential
        self.adjusted = adjusted

    def step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def adjust(self,
               x_proposed: torch.Tensor,
               x_current: torch.Tensor,
               log_alpha: torch.Tensor) -> torch.Tensor:
        # Apply "accept/reject" step (adjustment).
        log_u = torch.log(torch.rand_like(log_alpha))
        x_new = torch.clone(x_current)
        mask = log_u < log_alpha
        x_new[mask] = x_proposed[mask]
        return x_new

    def adapt(self):
        # optionally adapt the kernel
        pass

    def sample(self,
               x0: torch.Tensor,
               full_output: bool = False,
               n_iterations: int = 1000):
        n_chains, n_dim = x0.shape
        x = deepcopy(x0).detach()

        if full_output:
            draws = torch.zeros(
                size=(n_iterations, n_chains, n_dim),
                dtype=torch.float,
                requires_grad=False
            )
            draws[0] = x0

        for i in range(1, n_iterations):
            x_proposed, log_alpha = self.step(x)
            x_proposed = x_proposed.detach()
            log_alpha = log_alpha.detach()
            if self.adjusted:
                x = self.adjust(x_proposed, x, log_alpha)
            else:
                x = x_proposed
            if full_output:
                draws[i] = torch.clone(x)
            self.adapt()

        if full_output:
            return draws
        else:
            return x
