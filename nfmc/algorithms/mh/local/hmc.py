import math
from typing import Tuple, Union
import torch

from nfmc.algorithms.mh.local.base import LocalMHKernel
from nfmc.util import compute_divergence_mask, grad_f, sum_except_batch


def hmc_step_b(x: torch.Tensor,
               momentum: torch.Tensor,
               step_size: torch.Tensor,
               neg_log_prob_target: callable,
               event_shape: Union[Tuple[int, ...], torch.Size],
               return_neg_log_prob_and_grad: bool = False):
    """
    HMC momentum update.

    :param torch.Tensor x: state tensor with shape `(n_chains, *event_shape)`.
    :param torch.Tensor momentum: momentum tensor with shape `(n_chains, *event_shape)`.
    :param torch.Tensor step_size: step size tensor with shape `(n_chains,)` or `()`.

    :return: transformed momentum, number of target density calls, number of target density gradient calls.
    """
    if len(step_size.shape) == 1:
        step_size = step_size.view(
            step_size.shape[0], *[1] * (len(x.shape) - 1))

    fval, g, nc, ng = grad_f(x, neg_log_prob_target, event_shape)
    new_momentum = momentum - step_size / 2 * g
    new_momentum = new_momentum.to(x)
    g = g.to(x)

    if return_neg_log_prob_and_grad:
        return new_momentum, nc, ng, fval, g
    else:
        return new_momentum, nc, ng


def hmc_step_a(x: torch.Tensor,
               momentum: torch.Tensor,
               step_size: torch.Tensor):
    """
    HMC position update.

    :param torch.Tensor x: state tensor with shape `(n_chains, *event_shape)`.
    :param torch.Tensor momentum: momentum tensor with shape `(n_chains, *event_shape)`.
    :param torch.Tensor step_size: step size tensor with shape `(n_chains,)` or `()`.

    :return: transformed position.
    """
    if len(step_size.shape) == 1:
        step_size = step_size.view(
            step_size.shape[0], *[1] * (len(x.shape) - 1))
    new_position = x + step_size * momentum
    new_position = new_position.to(x)
    return new_position


def leapfrog_step(x: torch.Tensor,
                  momentum: torch.Tensor,
                  event_shape: Union[Tuple[int, ...], torch.Size],
                  step_size: torch.Tensor,
                  neg_log_prob_target: callable,
                  return_neg_log_prob_and_grad: bool = False):
    """
    Perform one leapfrog step.

    :param bool return_neg_log_prob_and_grad: if True, return the negative log probability
     and its gradient from the second B step.
    """
    n_calls = 0
    n_grads = 0

    momentum, nc, ng = hmc_step_b(
        x,
        momentum,
        step_size,
        neg_log_prob_target,
        event_shape
    )
    n_calls += nc
    n_grads += ng

    x = hmc_step_a(x, momentum, step_size)

    momentum, nc, ng, fval, g = hmc_step_b(
        x,
        momentum,
        step_size,
        neg_log_prob_target,
        event_shape,
        return_neg_log_prob_and_grad=True
    )
    n_calls += nc
    n_grads += ng

    if return_neg_log_prob_and_grad:
        return x, momentum, n_calls, n_grads, fval, g
    else:
        return x, momentum, n_calls, n_grads


def hmc_trajectory(x: torch.Tensor,
                   momentum: torch.Tensor,
                   event_shape: Union[Tuple[int, ...], torch.Size],
                   step_size: torch.Tensor,
                   n_leapfrog_steps: int,
                   neg_log_prob_target: callable,
                   full_output: bool = False):
    """
    Simulates a HMC trajectory with the Leapfrog integrator.

    :param torch.Tensor x: position tensor with shape `(n_chains, *event_shape)`.
    :param torch.Tensor x: momentum tensor with shape `(n_chains, *event_shape)`.
    :param Union[Tuple[int, ...], torch.Size] event_shape: event shape of the input tensor.
    :param torch.Tensor step_size: step size tensor with shape `(n_chains,)` or `()`.
    :param int n_leapfrog_steps: number of leapfrog steps, i.e., the trajectory length.
    :param callable neg_log_prob_target: function that computes the negative log target probability density.
    :param bool full_output: if True, return the entire trajectory as the output tuple element.
    :return: final position and momentum tensors, each with shape `(n_chains, *event_shape)`, as well as the total 
     numbers of target density evaluations and target density gradient evaluations.
    """
    full_trajectory = []

    n_calls = 0
    n_grads = 0

    for _ in range(n_leapfrog_steps):
        x, momentum, n_new_calls, n_new_grads = leapfrog_step(
            x=x,
            momentum=momentum,
            event_shape=event_shape,
            step_size=step_size,
            neg_log_prob_target=neg_log_prob_target
        )
        n_calls += n_new_calls
        n_grads += n_new_grads

        if full_output:
            full_trajectory.append(x)
    if full_output:
        return x, momentum, n_calls, n_grads, torch.stack(full_trajectory)
    return x, momentum, n_calls, n_grads


class HMCKernel(LocalMHKernel):
    """
    HMC kernel.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 neg_log_prob_target: callable,
                 n_leapfrog_steps: int = 20,
                 target_acceptance_rate: float = 0.651,
                 **kwargs):
        """
        HMCKernel constructor.

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param callable neg_log_prob_target: negative log probability density function. 
         Takes as input an event tensor with shape `(n_chains, *event_shape)` and outputs 
         a negative log probability tensor with shape `(n_chains,)`.
        :param int n_leapfrog_steps: number of leapfrog steps in each trajectory.
        :param kwargs: keyword arguments for the LocalMHKernel constructor.
        """
        super().__init__(
            event_shape,
            neg_log_prob_target,
            target_acceptance_rate=target_acceptance_rate,
            **kwargs
        )
        self.n_leapfrog_steps = n_leapfrog_steps

    @property
    def name(self):
        return 'HMC'

    def step(self,
             x: torch.Tensor,
             update: bool = False) -> torch.Tensor:
        """
        Perform one HMC transition.

        :param torch.Tensor x: incoming state tensor with shape `(*batch_shape, *event_shape)`.
        :param bool update: if True, update kernel parameters.
        :return: new state tensor with shape `(*batch_shape, *event_shape)`.
        """
        if self.warmup_active:
            step_size = self._dual_averaging.value
        else:
            step_size = self.step_size

        # Sample momentum and simulate trajectory
        p = torch.randn_like(x)
        x_prime, p_prime, nc, ng = hmc_trajectory(
            x=x.clone(),
            momentum=p,
            event_shape=self.event_shape,
            step_size=step_size,
            n_leapfrog_steps=self.n_leapfrog_steps,
            neg_log_prob_target=self.neg_log_prob_target
        )
        self.increment_n_calls(nc)
        self.increment_n_grads(ng)

        # Compute divergence mask
        divergence_mask_x = compute_divergence_mask(x_prime, self.event_shape)
        divergence_mask_p = compute_divergence_mask(p_prime, self.event_shape)
        divergence_mask = divergence_mask_x | divergence_mask_p

        n_valid_proposals = int((~divergence_mask).long().sum())
        self.increment_n_divergences(int(divergence_mask.long().sum()))
        self.increment_n_divergences_per_chain(divergence_mask)

        # Compute acceptance mask
        acceptance_mask = torch.zeros_like(divergence_mask)
        if n_valid_proposals > 0:
            hamiltonian_start: torch.Tensor = (
                self.neg_log_prob_target(x[~divergence_mask])
                + 0.5 * sum_except_batch(
                    p[~divergence_mask] ** 2,
                    self.event_shape
                )
            )
            self.increment_n_calls(n_valid_proposals)

            hamiltonian_end: torch.Tensor = (
                self.neg_log_prob_target(x_prime[~divergence_mask])
                + 0.5 * sum_except_batch(
                    p_prime[~divergence_mask] ** 2,
                    self.event_shape
                )
            )
            self.increment_n_calls(n_valid_proposals)

            log_prob_accept = -hamiltonian_end - (-hamiltonian_start)
            log_u = torch.rand_like(log_prob_accept).log()
            acceptance_mask[~divergence_mask] = (log_u < log_prob_accept)
        x[acceptance_mask] = x_prime[acceptance_mask]
        x = x.detach().clone()

        if update:
            self._update(acceptance_mask)

        self.increment_n_steps()
        self.increment_n_attempted_transitions(n_chains=x.shape[0])
        self.increment_n_accepted_transitions(
            int(acceptance_mask.long().sum()))

        return x

    def __repr__(self):
        return (f'log step: {float(math.log(float(self.step_size))):.2f}, '
                f'leapfrogs: {self.n_leapfrog_steps}')
