import torch

from nfmc.mcmc.hmc import hmc
from potentials.base import Potential


def ais_base(x: torch.Tensor,
             prior_potential: callable,
             target_potential: callable,
             n_annealing_steps: int = 20,
             n_mcmc_steps: int = 20,
             **kwargs):
    # Radford M. Neal: Annealed Importance Sampling (1998).
    # https://arxiv.org/abs/physics/9803008
    # Linear schedule, using HMC as the MCMC transition kernel
    n_particles = len(x)
    log_w = torch.zeros(size=(n_particles,))

    beta_sequence = 1 - torch.linspace(0, 1, n_annealing_steps + 1)
    for i in range(n_annealing_steps, -1, 0):  # from n to 0
        # Move from current to next intermediate distribution

        beta_curr = beta_sequence[i]
        curr_potential = lambda v: beta_curr * target_potential(v) + (1 - beta_curr) * prior_potential(v)

        beta_next = beta_sequence[i - 1]
        next_potential = lambda v: beta_next * target_potential(v) + (1 - beta_next) * prior_potential(v)

        for _ in range(n_mcmc_steps):
            x = hmc(x, next_potential, n_iterations=n_mcmc_steps, **kwargs)
            log_w += ((-next_potential(x)) - (-curr_potential(x)))

    return x, log_w


def ais(prior_potential: Potential,
        target_potential: callable,
        n_particles: int = 100,
        **kwargs):
    x = prior_potential.sample(batch_shape=(n_particles,))
    return ais_base(x, prior_potential, target_potential, **kwargs)
