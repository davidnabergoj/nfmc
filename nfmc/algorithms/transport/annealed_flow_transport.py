import math
from copy import deepcopy
from typing import List

from tqdm import tqdm
import torch
import torch.optim as optim

from torchflows import Flow
from torchflows.bijections.base import Bijection

from nfmc.algorithms.sampling.mcmc import mh


def compute_smc_flow_step_quantities(x: torch.Tensor,
                                     x_tilde: torch.Tensor,
                                     log_W: torch.Tensor,
                                     log_det: torch.Tensor,
                                     prev_potential: callable,
                                     next_potential: callable):
    log_G = next_potential(x_tilde) + log_det - prev_potential(x)
    log_w = torch.logaddexp(log_W, log_G)
    log_W = log_w - torch.logsumexp(log_w, dim=0)
    return {
        'log_W': log_W,
        'log_w': log_w
    }


def detach_all(*args: torch.Tensor):
    for tensor in args:
        tensor.detach_()


def resample(x_transformed, log_w):
    n_particles = len(x_transformed)
    cat_dist = torch.distributions.Categorical(logits=log_w)
    resampled_indices = cat_dist.sample(sample_shape=torch.Size((n_particles,))).long()
    return deepcopy(x_transformed[resampled_indices])


def smc_flow_step(x_base: torch.Tensor,
                  x_train: torch.Tensor,
                  x_val: torch.Tensor,
                  bijection: Bijection,
                  prev_potential: callable,
                  next_potential: callable,
                  log_W_base: torch.Tensor,
                  log_W_train: torch.Tensor,
                  log_W_val: torch.Tensor,
                  sampling_threshold: float,
                  mcmc_sampler: callable = None):
    """Apply a sequential Monte Carlo temperature transition step with a normalizing flow.
    This means first applying the normalizing flow forward map and then correcting the resulting particle distribution
    with a MCMC procedure.

    We transform base, training, and validation particles with NF and MCMC as the transformed versions are needed in the
    remainder of the procedure.
    We use base particles to estimate the change in log Z, as it does not affect anything else in the procedure.
    We use training particles to estimate log ESS, as it decides if we resample or not.
    """
    n_base = len(x_base)
    n_train = len(x_train)
    n_val = len(x_val)

    x_base_transformed, log_det_base = bijection.forward(x_base)
    x_train_transformed, log_det_train = bijection.forward(x_train)
    x_val_transformed, log_det_val = bijection.forward(x_val)
    detach_all(
        x_base_transformed,
        x_train_transformed,
        x_val_transformed,
        log_det_base,
        log_det_train,
        log_det_val
    )

    data_base = compute_smc_flow_step_quantities(
        x_base,
        x_base_transformed,
        log_W_base,
        log_det_base,
        prev_potential,
        next_potential
    )
    data_train = compute_smc_flow_step_quantities(
        x_train,
        x_train_transformed,
        log_W_train,
        log_det_train,
        prev_potential,
        next_potential
    )
    data_val = compute_smc_flow_step_quantities(
        x_val,
        x_val_transformed,
        log_W_val,
        log_det_val,
        prev_potential,
        next_potential
    )

    log_ess_train = -torch.logsumexp(2 * data_train['log_w'], dim=0)
    log_ess_base = -torch.logsumexp(2 * data_base['log_w'], dim=0)  # Only for monitoring
    delta_log_Z = torch.sum(data_base['log_w'])

    if log_ess_train - math.log(n_train) <= math.log(sampling_threshold):
        x_base_resampled = resample(x_base_transformed, data_base['log_w'])
        x_train_resampled = resample(x_train_transformed, data_train['log_w'])
        x_val_resampled = resample(x_val_transformed, data_val['log_w'])

        log_W_base = torch.full(size=(n_base,), fill_value=math.log(1 / n_base))
        log_W_train = torch.full(size=(n_train,), fill_value=math.log(1 / n_train))
        log_W_val = torch.full(size=(n_val,), fill_value=math.log(1 / n_val))
    else:
        x_base_resampled = deepcopy(x_base_transformed)
        x_train_resampled = deepcopy(x_train_transformed)
        x_val_resampled = deepcopy(x_val_transformed)

    if mcmc_sampler is None:
        mcmc_sampler = mh

    x_resampled = torch.concat([x_base_resampled, x_train_resampled, x_val_resampled], dim=0)
    x_corrected = mcmc_sampler(
        x0=x_resampled,
        potential=next_potential,
        full_output=False
    )
    x_base_corrected = deepcopy(x_corrected[:n_base])
    x_train_corrected = deepcopy(x_corrected[n_base:n_base + n_train])
    x_val_corrected = deepcopy(x_corrected[n_base + n_train:])

    return {
        'x_base_corrected': x_base_corrected,
        'x_train_corrected': x_train_corrected,
        'x_val_corrected': x_val_corrected,
        'x_base_resampled': x_base_resampled,
        'x_train_resampled': x_train_resampled,
        'x_val_resampled': x_val_resampled,
        'x_base_transformed': x_base_transformed,
        'x_train_transformed': x_train_transformed,
        'x_val_transformed': x_val_transformed,
        'log_W_base': log_W_base,
        'log_W_train': log_W_train,
        'log_W_val': log_W_val,
        'delta_log_Z': delta_log_Z,
        'log_ess_train': log_ess_train,
        'log_ess_base': log_ess_base
    }


def annealed_flow_transport_base(prior_potential: callable,
                                 target_potential: callable,
                                 prior_potential_sample: callable,
                                 target_potential_sample: callable,
                                 flow: Flow,
                                 n_particles: int = 100,
                                 n_train_particles: int = 100,
                                 n_val_particles: int = 100,
                                 n_steps: int = 20,
                                 sampling_threshold: float = None,
                                 mcmc_sampler: callable = None,
                                 show_progress: bool = False,
                                 full_output: bool = False):
    """Annealed flow transport algorithm.
    This algorithm transports particles drawn from a prior distribution to a target distribution in a sequential Monte
     Carlo scheme.
    To transport particles between consecutive temperature levels, we first apply the forward map of a
     normalizing flow, then apply a correction with a Markov Chain Monte Carlo sampler.
    We use a linear schedule for temperature levels and Hamiltonian Monte Carlo for sample correction.

    :param prior_potential: potential function corresponding to the prior distribution.
    :param target_potential: potential function corresponding to the target distribution.
    :param flow: normalizing flow object.
    :param n_particles: number of particles to transport.
    :param n_steps: number of temperature levels.
    :param sampling_threshold: threshold that determines when to apply importance resampling.
     Must be between 1 / n_particles and 1.
    :param mcmc_sampler: sampler to use as MCMC correction.
    :param show_progress: if True, display a progress bar.
    :param full_output: if True, return draws at each temperature level instead of only the last one.
    :return: draws from the final temperature level with shape (n_particles, n_dim).
    """
    n_base_particles = n_particles
    assert n_base_particles > 1
    assert n_train_particles > 1
    assert n_val_particles > 1

    if sampling_threshold is None:
        # Try setting to 0.3
        if 1 / n_particles <= 0.3:
            sampling_threshold = 0.3
        else:
            sampling_threshold = 1 / n_base_particles
    assert 1 / n_base_particles <= sampling_threshold < 1

    x_base = prior_potential_sample(batch_shape=(n_base_particles,))
    x_train = prior_potential_sample(batch_shape=(n_train_particles,))
    x_val = prior_potential_sample(batch_shape=(n_val_particles,))
    log_W_base = torch.full(size=(n_base_particles,), fill_value=math.log(1 / n_base_particles))
    log_W_train = torch.full(size=(n_train_particles,), fill_value=math.log(1 / n_train_particles))
    log_W_val = torch.full(size=(n_val_particles,), fill_value=math.log(1 / n_val_particles))
    log_Z = 0.0

    if full_output:
        n_dim = x_base.shape[1]
        transformed_history = torch.zeros(
            size=(n_steps - 1, n_base_particles, n_dim),
            dtype=torch.float,
            requires_grad=False
        )
        resampled_history = torch.zeros(
            size=(n_steps - 1, n_base_particles, n_dim),
            dtype=torch.float,
            requires_grad=False
        )
        corrected_history = torch.zeros(
            size=(n_steps - 1, n_base_particles, n_dim),
            dtype=torch.float,
            requires_grad=False
        )
        x_base_initial = deepcopy(x_base.detach().clone())

    if show_progress:
        iterator = tqdm(range(1, n_steps), desc='AFT')
    else:
        iterator = range(1, n_steps)

    for k in iterator:
        prev_inv_temperature = (k - 1) / (n_steps - 1)
        curr_inv_temperature = k / (n_steps - 1)

        u_prev = lambda v: (1 - prev_inv_temperature) * prior_potential(v) + prev_inv_temperature * target_potential(v)
        u_next = lambda v: (1 - curr_inv_temperature) * prior_potential(v) + curr_inv_temperature * target_potential(v)

        with torch.enable_grad():
            # We could fit the NF with importance weights here!
            flow.base_log_prob = lambda x: -u_next(x)  # Overwrite the base log probability
            flow.fit(x_train, x_val=x_val, early_stopping=True)

        outputs = smc_flow_step(
            x_base=x_base,
            x_train=x_train,
            x_val=x_val,
            bijection=flow.bijection,
            prev_potential=u_prev,
            next_potential=u_next,
            log_W_base=log_W_base,
            log_W_train=log_W_train,
            log_W_val=log_W_val,
            sampling_threshold=sampling_threshold,
            mcmc_sampler=mcmc_sampler
        )
        x_base = outputs['x_base_corrected']
        x_train = outputs['x_train_corrected']
        x_val = outputs['x_val_corrected']
        log_W_base = outputs['log_W_base']
        log_W_train = outputs['log_W_train']
        log_W_val = outputs['log_W_val']
        log_Z += outputs['delta_log_Z']

        if full_output:
            transformed_history[k - 1] = outputs['x_base_transformed']
            resampled_history[k - 1] = outputs['x_base_resampled']
            corrected_history[k - 1] = outputs['x_base_corrected']

        if show_progress:
            iterator.set_postfix_str(
                f'Temperature: {(1 - curr_inv_temperature):.2f}, '
                f'log Z: {log_Z:.4f}, '
                f'Base log ESS: {outputs["log_ess_base"]:.4f}, '
                f'Train log ESS: {outputs["log_ess_train"]:.4f}'
            )

    if full_output:
        outputs = torch.zeros(
            size=(1 + 3 * (n_steps - 1), n_base_particles, n_dim),
            dtype=torch.float,
            requires_grad=False
        )
        outputs[0] = x_base_initial

        transformed_idx = torch.arange(1, len(outputs), 3)
        outputs[transformed_idx] = transformed_history

        resampled_idx = torch.arange(2, len(outputs), 3)
        outputs[resampled_idx] = resampled_history

        corrected_idx = torch.arange(3, len(outputs), 3)
        outputs[corrected_idx] = corrected_history

        return outputs
    return x_base


def continual_repeated_annealed_flow_transport_base(prior_potential: callable,
                                                    target_potential: callable,
                                                    prior_potential_sample: callable,
                                                    target_potential_sample: callable,
                                                    bijections: List[Bijection],
                                                    n_particles: int = 100,
                                                    n_training_steps: int = 100,
                                                    n_annealing_steps: int = 20,
                                                    sampling_threshold: float = 0.3,
                                                    full_output: bool = False,
                                                    show_progress: bool = False):
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

    if show_progress:
        iterator = tqdm(range(1, n_training_steps), desc='CRAFT')
    else:
        iterator = range(1, n_training_steps)

    # Train
    for j in iterator:
        x = prior_potential_sample(batch_shape=(n_particles,))
        log_W = torch.full(size=(n_particles,), fill_value=-math.log(n_particles))
        log_Z = 0.0
        for k in range(1, n_annealing_steps + 1):
            prev_lambda = (k - 1) / n_annealing_steps
            prev_potential = lambda v: (1 - prev_lambda) * prior_potential(v) + prev_lambda * target_potential(v)

            next_lambda = k / n_annealing_steps
            next_potential = lambda v: (1 - next_lambda) * prior_potential(v) + next_lambda * target_potential(v)

            with torch.enable_grad():
                optimizers[k - 1].zero_grad()
                loss_value = loss(x, log_W.exp(), bijections[k - 1], prev_potential, next_potential)
                loss_value.backward()
                optimizers[k - 1].step()

            with torch.no_grad():
                x, log_Z, log_W = smc_flow_step(
                    x=x,
                    bijection=bijections[k - 1],
                    prev_potential=prev_potential,
                    next_potential=next_potential,
                    log_W=log_W,
                    log_Z=log_Z,
                    sampling_threshold=sampling_threshold
                )

    # Sample
    particle_history = []
    with torch.no_grad():
        x = prior_potential_sample(batch_shape=(n_particles,))
        particle_history.append(x)
        log_W = torch.full(size=(n_particles,), fill_value=-math.log(n_particles))
        log_Z = 0.0
        for k in range(1, n_annealing_steps + 1):
            prev_lambda = (k - 1) / n_annealing_steps
            prev_potential = lambda v: (1 - prev_lambda) * prior_potential(v) + prev_lambda * target_potential(v)

            next_lambda = k / n_annealing_steps
            next_potential = lambda v: (1 - next_lambda) * prior_potential(v) + next_lambda * target_potential(v)

            x, log_Z, log_W = smc_flow_step(
                x=x,
                bijection=bijections[k - 1],
                prev_potential=prev_potential,
                next_potential=next_potential,
                log_W=log_W,
                log_Z=log_Z,
                sampling_threshold=sampling_threshold
            )
            particle_history.append(x)

    particle_history = torch.stack(particle_history)

    if full_output:
        return particle_history, bijections, log_Z
    return particle_history
