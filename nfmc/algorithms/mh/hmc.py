import math
from typing import Optional, Tuple, Union
import torch

from nfmc.algorithms.mh.base import MHSampler, MHKernel
from nfmc.util import diag_mult, grad_f, sum_except_batch


def hmc_step_b(x: torch.Tensor,
               momentum: torch.Tensor,
               step_size: float,
               neg_log_prob_target: callable,
               event_shape: Union[Tuple[int, ...], torch.Size]):
    """
    HMC momentum update.

    :return: transformed momentum, number of target density calls, number of target density gradient calls.
    """
    _, g, nc, ng = grad_f(x, neg_log_prob_target, event_shape)
    return momentum - step_size / 2 * g, nc, ng


def hmc_step_a(x: torch.Tensor,
               momentum: torch.Tensor,
               inv_mass_diag,
               step_size: float,
               event_shape: Union[Tuple[int, ...], torch.Size]):
    """
    HMC position update.

    :return: transformed position.
    """
    return x + step_size * diag_mult(momentum, inv_mass_diag.to(momentum), event_shape)


def hmc_trajectory(x: torch.Tensor,
                   momentum: torch.Tensor,
                   event_shape: Union[Tuple[int, ...], torch.Size],
                   step_size: float,
                   n_leapfrog_steps: int,
                   inv_mass_diag: torch.Tensor,
                   neg_log_prob_target: callable,
                   full_output: bool = False):
    """
    Simulates a HMC trajectory with the Leapfrog integrator.

    :param torch.Tensor x: position tensor with shape `(n_chains, *event_shape)`.
    :param torch.Tensor x: momentum tensor with shape `(n_chains, *event_shape)`.
    :param Union[Tuple[int, ...], torch.Size] event_shape: event shape of the input tensor.
    :param float step_size: HMC step size.
    :param int n_leapfrog_steps: number of leapfrog steps, i.e., the trajectory length.
    :param torch.Tensor inv_mass_diag: diagonal of the inverse mass matrix with shape `(event_size,)`, where 
     `event_size` is the number of event elements.
    :param callable neg_log_prob_target: function that computes the negative log target probability density.
    :param bool full_output: if True, return the entire trajectory as the output tuple element.
    :return: final position and momentum tensors, each with shape `(n_chains, *event_shape)`, as well as the total 
     numbers of target density evaluations and target density gradient evaluations.
    """
    full_trajectory = []

    n_calls = 0
    n_grads = 0

    for _ in range(n_leapfrog_steps):
        momentum, nc, ng = hmc_step_b(
            x,
            momentum,
            step_size,
            neg_log_prob_target,
            event_shape
        )
        n_calls += nc
        n_grads += ng

        x = hmc_step_a(x, momentum, inv_mass_diag, step_size, event_shape)

        momentum, nc, ng = hmc_step_b(
            x,
            momentum,
            step_size,
            neg_log_prob_target,
            event_shape
        )
        n_calls += nc
        n_grads += ng

        if full_output:
            full_trajectory.append(x)
    if full_output:
        return x, momentum, n_calls, n_grads, torch.stack(full_trajectory)
    return x, momentum, n_calls, n_grads


class HMCKernel(MHKernel):
    """
    HMC kernel with a diagonal mass matrix.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 neg_log_prob_target: callable,
                 step_size: float = 0.01,
                 n_leapfrog_steps: int = 20,
                 inv_mass_diag: Optional[torch.Tensor] = None):
        super().__init__(event_shape, neg_log_prob_target)

        self.step_size = step_size
        self.n_leapfrog_steps = n_leapfrog_steps
        self.inv_mass_diag = inv_mass_diag
        if self.inv_mass_diag is None:
            self.inv_mass_diag = torch.ones(
                size=(self.event_size,), dtype=torch.double)

    def step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Sample momentum
        noise = torch.randn_like(x)
        p = diag_mult(
            noise, 
            1 / self.kernel.inv_mass_diag.sqrt().to(x), 
            self.event_shape
        )

        # Simulate trajectory
        x_prime, p_prime, nc, ng = hmc_trajectory(
            x=x,
            p=p,
            event_shape=self.event_shape,
            step_size=self.step_size,
            n_leapfrog_steps=self.n_leapfrog_steps,
            inv_mass_diag=self.inv_mass_diag,
            neg_log_prob_target=self.neg_log_prob_target
        )
        self._n_calls += nc
        self._n_grads += ng

        # Compute divergence mask
        divergence_mask_x = sum_except_batch(
            (~torch.isfinite(x_prime)).long(), self.event_shape
        ) > 0
        divergence_mask_p = sum_except_batch(
            (~torch.isfinite(p_prime)).long(), self.event_shape
        ) > 0
        divergence_mask = divergence_mask_x | divergence_mask_p

        # Compute acceptance mask
        acceptance_mask = torch.zeros_like(divergence_mask)
        n_valid = int(torch.sum((~divergence_mask).long()))
        if n_valid > 0:
            hamiltonian_start: torch.Tensor = self.neg_log_prob_target(x[~divergence_mask]) + 0.5 * sum_except_batch(
                diag_mult(
                    p[~divergence_mask] ** 2, self.kernel.inv_mass_diag, self.event_shape
                ),
                self.event_shape
            )
            self._n_calls += n_valid

            hamiltonian_end: torch.Tensor = self.neg_log_prob_target(x_prime[~divergence_mask]) + 0.5 * sum_except_batch(
                diag_mult(
                    p_prime[~divergence_mask] ** 2, self.kernel.inv_mass_diag, self.event_shape
                ),
                self.event_shape
            )
            self._n_calls += n_valid

            log_prob_accept = -hamiltonian_end - (-hamiltonian_start)
            log_u = torch.rand_like(log_prob_accept).log()
            acceptance_mask[~divergence_mask] = (log_u < log_prob_accept)

        return x_prime.detach(), acceptance_mask

    def __repr__(self):
        return (f'log step: {math.log(self.step_size):.2f}, '
                f'leapfrogs: {self.n_leapfrog_steps}, '
                f'mass norm: {torch.max(torch.abs(self.inv_mass_diag)):.2f}')
