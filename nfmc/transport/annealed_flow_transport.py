import math
from copy import deepcopy

import torch
import torch.optim as optim

from nfmc.mcmc.hmc import hmc_trajectory
from normalizing_flows import Flow
from normalizing_flows.bijections import RealNVP
from normalizing_flows.bijections.base import Bijection
from potentials.base import Potential


def smc_flow_step(x, flow, prev_potential, next_potential, log_W, log_Z, sampling_threshold):
    n_particles, n_dim = x.shape
    x_tilde, log_det = flow.bijection.forward(x)

    # next_lambda = k / n_steps
    # next_potential = lambda v: (1 - next_lambda) * prior_potential(v) + next_lambda * target_potential(v)
    #
    # prev_lambda = (k - 1) / n_steps
    # prev_potential = lambda v: (1 - prev_lambda) * prior_potential(v) + prev_lambda * target_potential(v)

    log_G = next_potential(x_tilde) + log_det - prev_potential(x)
    log_w = torch.logaddexp(log_W, log_G)
    log_Z += torch.sum(log_w)

    log_W = log_w - torch.logsumexp(log_w, dim=0)

    # ess = 1 / sum(exp(2 * log_w))
    # log_ess = -log(sum(exp(2*log_w)))
    log_ess = - torch.logsumexp(2 * log_w, dim=0)

    if log_ess - math.log(n_particles) <= math.log(sampling_threshold):
        cat_dist = torch.distributions.Categorical(logits=log_w)
        resampled_indices = cat_dist.sample(sample_shape=torch.Size((n_particles,))).long()
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

    return x, log_Z, log_W


def annealed_flow_transport_base(prior_potential: Potential,
                                 target_potential: Potential,
                                 flow: Flow,
                                 n_particles: int = 100,
                                 n_steps: int = 20,
                                 sampling_threshold: float = None,
                                 full_output: bool = False):
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
    assert n_particles > 1

    if sampling_threshold is None:
        # Try setting to 0.3
        if 1 / n_particles <= 0.3:
            sampling_threshold = 0.3
        else:
            sampling_threshold = 1 / n_particles

    assert 1 / n_particles <= sampling_threshold < 1
    x = prior_potential.sample(batch_shape=(n_particles,))
    log_W = torch.full(size=(n_particles,), fill_value=math.log(1 / n_particles))
    log_Z = 0.0

    xs = [deepcopy(x.detach())]
    for k in range(1, n_steps):
        with torch.enable_grad():
            flow.fit(x)
        with torch.no_grad():
            prev_lambda = (k - 1) / n_steps
            next_lambda = k / n_steps
            x, log_Z, log_W = smc_flow_step(
                x=x,
                flow=flow,
                prev_potential=lambda v: (1 - prev_lambda) * prior_potential(v) + prev_lambda * target_potential(v),
                next_potential=lambda v: (1 - next_lambda) * prior_potential(v) + next_lambda * target_potential(v),
                log_W=log_W,
                log_Z=log_Z,
                sampling_threshold=sampling_threshold
            )
            xs.append(deepcopy(x.detach()))

    if full_output:
        return torch.stack(xs)
    return x


def continual_repeated_annealed_flow_transport_base(prior_potential: Potential,
                                                    target_potential: Potential,
                                                    bijections: list[Bijection],
                                                    n_particles: int = 100,
                                                    n_training_steps: int = 100,
                                                    n_annealing_steps: int = 20,
                                                    sampling_threshold: float = 0.3,
                                                    full_output: bool = False):
    """
    Idea:
    * have a flow for each annealing step that connects the intermediate potentials
    * the last flow approximates the target
    * train these flows in J steps

    :param n_particles:
    :param n_annealing_steps: number of annealing steps/levels.
    :return:
    """
    assert 1 / n_particles <= sampling_threshold < 1
    assert len(bijections) == n_annealing_steps
    n_dim = prior_potential.event_shape[0]
    optimizers = [optim.AdamW(flow.parameters()) for flow in bijections]

    def loss(x_prev: torch.Tensor,
             W_prev: torch.Tensor,  # Not the log!!!
             bijection: Bijection,
             u_prev: callable,
             u_next: callable):
        x_next, log_det = bijection.inverse(x_prev)
        d = -u_prev(x_prev) + u_next(x_next) + log_det  # (n_particles,)
        loss = torch.sum(W_prev * d)
        return loss

    # Train
    for j in range(n_training_steps):
        x = prior_potential.sample(batch_shape=(n_particles,))
        log_W = torch.full(size=(n_particles,), fill_value=-math.log(n_particles))
        log_Z = 0.0
        for k in range(n_annealing_steps):
            prev_lambda = (k - 1) / n_annealing_steps
            prev_potential = lambda v: (1 - prev_lambda) * prior_potential(v) + prev_lambda * target_potential(v)

            next_lambda = k / n_annealing_steps
            next_potential = lambda v: (1 - next_lambda) * prior_potential(v) + next_lambda * target_potential(v)

            optimizers[k].zero_grad()
            loss_value = loss(x, log_W.exp(), bijections[k], prev_potential, next_potential)
            loss_value.backward()

            x, log_Z, log_W = smc_flow_step(
                x=x,
                flow=bijections[k],
                prev_potential=prev_potential,
                next_potential=next_potential,
                log_W=log_W,
                log_Z=log_Z,
                sampling_threshold=sampling_threshold
            )
            optimizers[k].step()

    # Sample
    x = prior_potential.sample(batch_shape=(n_particles,))
    log_W = torch.full(size=(n_particles,), fill_value=-math.log(n_particles))
    log_Z = 0.0
    for k in range(n_annealing_steps):
        prev_lambda = (k - 1) / n_annealing_steps
        prev_potential = lambda v: (1 - prev_lambda) * prior_potential(v) + prev_lambda * target_potential(v)

        next_lambda = k / n_annealing_steps
        next_potential = lambda v: (1 - next_lambda) * prior_potential(v) + next_lambda * target_potential(v)

        x, log_Z, log_W = smc_flow_step(
            x=x,
            flow=bijections[k],
            prev_potential=prev_potential,
            next_potential=next_potential,
            log_W=log_W,
            log_Z=log_Z,
            sampling_threshold=sampling_threshold
        )

    if full_output:
        return x, bijections, log_Z
    return x
