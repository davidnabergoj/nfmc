import math
from copy import deepcopy

import torch

from nfmc.mcmc.hmc import hmc_trajectory
from normalizing_flows import Flow
from potentials.base import Potential


def aft(
        prior_potential: Potential,
        target_potential: Potential,
        flow: Flow,
        n_particles: int = 100,
        n_steps: int = 20,
        sampling_threshold: float = 0.3,
        full_output: bool = False
):
    """
    Linear annealing schedule.
    Using HMC kernel.

    :param prior_potential:
    :param target_potential:
    :param flow:
    :param n_particles:
    :param n_steps:
    :param sampling_threshold:
    :return:
    """
    assert 1 / n_particles <= sampling_threshold < 1
    x = prior_potential.sample(batch_shape=(n_particles,))
    n_dim = prior_potential.event_shape[0]  # TODO add support for general event shapes
    log_W = torch.full(size=(n_particles,), fill_value=math.log(1 / n_particles))
    log_Z = 0.0

    # TODO check shapes

    xs = [deepcopy(x.detach())]
    for k in range(1, n_steps):
        flow.fit(x)
        x_tilde, log_det = flow.bijection.forward(x)

        next_lambda = k / n_steps
        next_potential = lambda v: (1 - next_lambda) * prior_potential(v) + next_lambda * target_potential(v)

        prev_lambda = (k - 1) / n_steps
        prev_potential = lambda v: (1 - prev_lambda) * prior_potential(v) + prev_lambda * target_potential(v)

        log_G = next_potential(x_tilde) + log_det - prev_potential(x)
        log_w = torch.logaddexp(log_W + log_G)
        log_Z += torch.sum(log_w)

        log_W = log_w - torch.logsumexp(log_w, dim=0)

        # ess = 1 / sum(exp(2 * log_w))
        # log_ess = -log(sum(exp(2*log_w)))
        log_ess = - torch.logsumexp(2 * log_w, dim=0)

        if log_ess - math.log(n_particles) <= math.log(sampling_threshold):
            resampled_indices = torch.distributions.Categorical(logits=log_w).sample(size=(n_particles,)).long()
            x = x_tilde[resampled_indices]
            log_W = torch.full(size=(n_particles,), fill_value=math.log(1 / n_particles))
        else:
            x = x_tilde  # Not explicitly stated in pseudocode, but probably true

        momentum = torch.randn_like(x)
        inv_mass_diag = torch.ones(size=(1, n_dim))
        trajectory_length = 10
        step_size = n_dim ** (-1 / 4)
        x, _ = hmc_trajectory(
            x=x,
            momentum=momentum,
            inv_mass_diag=inv_mass_diag,
            n_leapfrog_steps=trajectory_length,
            step_size=step_size,
            potential=next_potential
        )
        xs.append(deepcopy(x.detach()))

    if full_output:
        return torch.stack(xs)
    return x
