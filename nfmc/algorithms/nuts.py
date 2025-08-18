import math
from typing import Tuple, Union
from dataclasses import dataclass

import torch
from nfmc.algorithms.kernel import MarkovKernel
from nfmc.util import sum_except_batch
from nfmc.algorithms.mh.local.hmc import leapfrog_step


@torch.no_grad()
def _kinetic_energy(p: torch.Tensor, event_shape: Union[Tuple[int, ...], torch.Size]) -> torch.Tensor:
    # Identity mass => 0.5 * p^T p
    return 0.5 * sum_except_batch((p * p), event_shape)


@dataclass
class ParallelTreeState:
    n_chains: int
    event_shape: Union[Tuple[int, ...], torch.Size]

    # Position variables
    x_minus: torch.Tensor = None
    x_plus: torch.Tensor = None

    # Momentum variables
    p_minus: torch.Tensor = None
    p_plus: torch.Tensor = None

    # Proposed state variables
    x_prime: torch.Tensor = None
    log_prob_prime: torch.Tensor = None

    # Other variables
    n_valid: torch.Tensor = None
    sum_accept_prob: torch.Tensor = None
    stop: torch.Tensor = None
    diverged: torch.Tensor = None
    n_leapfrogs: torch.Tensor = None

    def __post_init__(self):
        self.x_minus = torch.zeros(size=(self.n_chains, *self.event_shape))
        self.x_plus = torch.zeros(size=(self.n_chains, *self.event_shape))

        self.p_minus = torch.zeros(size=(self.n_chains, *self.event_shape))
        self.p_plus = torch.zeros(size=(self.n_chains, *self.event_shape))

        self.x_prime = torch.zeros(size=(self.n_chains, *self.event_shape))
        self.log_prob_prime = torch.zeros(size=(self.n_chains,))

        self.n_valid = torch.zeros(size=(self.n_chains,), dtype=torch.long)
        self.sum_accept_prob = torch.zeros(size=(self.n_chains,))
        self.stop = torch.zeros(size=(self.n_chains,), dtype=torch.bool)
        self.diverged = torch.zeros(size=(self.n_chains,), dtype=torch.bool)
        self.n_leapfrogs = torch.zeros(size=(self.n_chains,))

    def masked_copy(self, mask: torch.Tensor):
        """
        Create a copy of this state, retaining only masked chains.
        """
        return ParallelTreeState(
            n_chains=int(mask.sum()),
            event_shape=self.event_shape,
            x_minus=self.x_minus[mask].clone(),
            x_plus=self.x_plus[mask].clone(),

            p_minus=self.p_minus[mask].clone(),
            p_plus=self.p_plus[mask].clone(),

            x_prime=self.x_prime[mask].clone(),
            log_prob_prime=self.log_prob_prime[mask].clone(),

            n_valid=self.n_valid[mask].clone(),
            sum_accept_prob=self.sum_accept_prob[mask].clone(),
            stop=self.stop[mask].clone(),
            diverged=self.diverged[mask].clone(),
            n_leapfrogs=self.n_leapfrogs[mask].clone(),
        )

    def overwrite_with(self, other_state, mask: torch.Tensor):
        """
        Overwrite this state's data with other state's data.
        The other state may have fewer chains. 
        The affected chains for this state are determined by the mask.
        """
        if not mask.shape == (self.n_chains,):
            raise ValueError(
                f"Shape of overwrite mask should be equal to {(self.n_chains,)}, but got {mask.shape}"
            )
        self.x_minus[mask] = other_state.x_minus
        self.x_plus[mask] = other_state.x_plus

        self.p_minus[mask] = other_state.p_minus
        self.p_plus[mask] = other_state.p_plus

        self.x_prime[mask] = other_state.x_prime
        self.log_prob_prime[mask] = other_state.log_prob_prime
        self.n_valid[mask] = other_state.n_valid
        self.sum_accept_prob[mask] = other_state.sum_accept_prob
        self.stop[mask] = other_state.stop
        self.diverged[mask] = other_state.diverged
        self.n_leapfrogs[mask] = other_state.n_leapfrogs


def _acceptance_prob(log_prob_new,
                     momentum_new,
                     log_prob_old,
                     momentum_old):
    h_new = -log_prob_new + _kinetic_energy(momentum_new)
    h_old = -log_prob_old + _kinetic_energy(momentum_old)
    return torch.clamp(torch.exp(h_old-h_new), max=1.0)


def is_uturn(x_minus,
             x_plus,
             p_minus,
             p_plus):
    dx = x_plus - x_minus
    m1 = torch.einsum('...i,...j->...', dx, p_minus) < 0
    m2 = torch.einsum('...i,...j->...', dx, p_plus) < 0
    return m1 | m2


def _build_tree(x: torch.Tensor,
                p: torch.Tensor,
                event_shape: Union[torch.Size, Tuple[int, ...]],
                u_slice: torch.Tensor,
                v: torch.Tensor,
                j: torch.Tensor,
                step_size: torch.Tensor,
                neg_log_prob_target: callable,
                log_prob_x: torch.Tensor,
                max_delta: float) -> ParallelTreeState:
    """
    Build a balanced binary tree.
    A separate tree is built individually for each chain state via vectorization.

    :param torch.Tensor x: position tensor with shape `(n_chains, *event_shape)`.
    :param torch.Tensor p: momentum tensor with shape `(n_chains, *event_shape)`.
    :param Union[torch.Size, Tuple[int, ...]] event_shape: shape of the position and momentum tensors.
    :param torch.Tensor u_slice: slice variable tensor with shape `(n_chains,)`.
    :param torch.Tensor v: direction tensor with shape `(n_chains,)`.
    :param torch.Tensor j: tree depth tensor with shape `(n_chains,)`.
    :param torch.Tensor step_size: step size tensor with shape `(n_chains,)`.
    :param callable neg_log_prob_target: negative log probability density function. 
        Takes as input an event tensor with shape `(n_chains, *event_shape)` and outputs 
        a negative log probability tensor with shape `(n_chains,)`.
    :param torch.Tensor log_prob_x: log probability density of incoming position tensor x with shape `(n_chains,)`.
    :param float max_delta: energy error divergence threshold.
    """

    # Clone position and momentum tensors
    state = ParallelTreeState(
        n_chains=x.shape[0],
        event_shape=event_shape
    )

    base_case_mask = (j == 0)
    general_case_mask = (j > 0)

    # Base case
    if base_case_mask.any():
        # Perform a single leapfrog step
        x1, p1, neg_log_prob1, g1 = leapfrog_step(
            x=x[base_case_mask],
            momentum=p[base_case_mask],
            event_shape=event_shape,
            step_size=v[base_case_mask] * step_size[base_case_mask],
            neg_log_prob_target=neg_log_prob_target,
            return_neg_log_prob_and_grad=True
        )
        log_prob1 = -neg_log_prob1

        joint = log_prob1 - _kinetic_energy(p1)
        diverged = ((joint - log_prob_x[base_case_mask])
                    < -max_delta) | (~torch.isfinite(log_prob1))
        valid = (torch.log(u_slice) <= joint) & (
            ~diverged) & torch.isfinite(log_prob1)

        state.x_minus[base_case_mask], state.p_minus[base_case_mask] = x1, p1
        state.x_plus[base_case_mask], state.p_plus[base_case_mask] = x1, p1
        state.x_prime[base_case_mask], state.log_prob_prime[base_case_mask] = x1, log_prob1

        state.n_valid[base_case_mask][valid] = 1
        # Redundant, but kept for safety
        state.n_valid[base_case_mask][~valid] = 0
        state.sum_accept_prob[base_case_mask] = _acceptance_prob(
            log_prob1,
            p1,
            log_prob_x[base_case_mask],
            p[base_case_mask]
        )
        state.n_leapfrogs[base_case_mask] = 1
        state.stop[base_case_mask] = diverged
        state.diverged[base_case_mask] = diverged

    # General case
    if general_case_mask.any():
        # Build the left subtree
        left = _build_tree(
            x=x[general_case_mask],
            p=p[general_case_mask],
            event_shape=event_shape,
            u_slice=u_slice[general_case_mask],
            v=v[general_case_mask],
            j=j[general_case_mask] - 1,
            step_size=step_size[general_case_mask],
            neg_log_prob_target=neg_log_prob_target,
            log_prob_x=log_prob_x[general_case_mask],
            max_delta=max_delta
        )

        # Early stop if left diverged or told to stop
        state_left_stop = left.masked_copy(left.stop)
        stop_copy_mask = general_case_mask.clone()
        stop_copy_mask[general_case_mask] = left.stop
        state.overwrite_with(state_left_stop, stop_copy_mask)

        state_left_continue = left.masked_copy(~left.stop)

        negative_mask = (v == -1)
        positive_mask = ~negative_mask

        x_start, p_start = x.clone(), p.clone()
        x_start[negative_mask], p_start[negative_mask] = state_left_continue.x_minus, state_left_continue.p_minus
        x_start[positive_mask], p_start[positive_mask] = state_left_continue.x_plus, state_left_continue.p_plus

        right = _build_tree(
            x=x_start,
            p=p_start,
            event_shape=event_shape,
            u_slice=u_slice[~left.stop],
            v=v[~left.stop],
            j=j[~left.stop] - 1,
            step_size=step_size[~left.stop],
            neg_log_prob_target=neg_log_prob_target,
            log_prob_x=log_prob_x[~left.stop],
            max_delta=max_delta
        )

        # Combine
        ...

        # Choose a proposal uniformly from valid points
        state.n_valid = left.n_valid + right.n_valid
        valid_mask = state.n_valid > 0

        # If1
        rand_mask = torch.less(
            torch.rand((state.n_chains,)),
            right.n_valid / state.n_valid
        )

        # If2
        state.x_prime[rand_mask], state.log_prob_prime[rand_mask] = right.x_prime[rand_mask], right.log_prob_prime[rand_mask]
        # Else2
        state[~rand_mask], state.log_prob_prime[~rand_mask] = left.x_prime[~rand_mask], left.log_prob_prime[~rand_mask]
        # Endif2

        # Else1
        state.x_prime[~valid_mask], state.log_prob_prime[~valid_mask] = left.x_prime[~valid_mask], left.log_prob_prime[~valid_mask]
        # EndIf1

    state.sum_accept_prob = left.sum_accept_prob + right.sum_accept_prob
    state.n_leapfrogs = left.n_leapfrogs + right.n_leapfrogs
    state.diverged = left.diverged | right.diverged

    state.stop = right.stop | is_uturn(
        left.x_minus,
        right.x_plus,
        left.p_minus,
        right.p_plus
    )

    return state


class NUTSKernel(MarkovKernel):
    """
    Implementation of the no-U-turn sampler (NUTS) transition kernel.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 neg_log_prob_target: callable,
                 max_tree_depth: int = 10,
                 max_delta: float = 1000.0,
                 **kwargs):
        """
        NUTSKernel constructor.

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param callable neg_log_prob_target: negative log probability density function. 
         Takes as input an event tensor with shape `(n_chains, *event_shape)` and outputs 
         a negative log probability tensor with shape `(n_chains,)`.
        :param int max_tree_depth: maximum depth of the balanced binary tree.
        :param float max_delta: maximum allowed value of the Hamiltonian dynamics energy error.
         If the simulated trajectory error exceeds this threshold, it is flagged as having diverged.
        """
        super().__init__(event_shape, neg_log_prob_target, **kwargs)
        self.max_tree_depth = max_tree_depth
        self.max_delta = max_delta

    @property
    def name(self):
        return 'NUTS'

    def step(self,
             x: torch.Tensor,
             update: bool = False) -> torch.Tensor:
        """
        Perform one NUTS transition.

        :param torch.Tensor x: incoming state tensor with shape `(*batch_shape, *event_shape)`.
        :param bool update: if True, update kernel parameters.
        :return: new state tensor with shape `(*batch_shape, *event_shape)`.
        """
        if self.warmup_active:
            step_size = self._dual_averaging.value
        else:
            step_size = self.step_size

        log_prob = -self.neg_log_prob_target(x)

        # Sample momentum and slice variable
        p0 = torch.randn_like(x)
        joint0 = float(log_prob - _kinetic_energy(p0))
        u_slice = torch.rand(()) * math.exp(joint0)

        # Initialize balanced binary tree
        x_minus, x_plus = x.clone(), x.clone()
        p_minus, p_plus = p0.clone(), p0.clone()
        x_prime = x.clone()
        log_prob_prime = log_prob.clone()

        for j in range(self.max_tree_depth + 1):
            # Choose direction
            if torch.randint(0, 2, ()).item() == 1:
                v = 1
            else:
                v = -1

            if v == -1:
                # Build tree into negative time direction
                state: TreeState = _build_tree()
                x_minus, p_minus = state.x_minus, state.p_minus
            else:
                # Build tree into positive time direction
                state: TreeState = _build_tree()
                x_plus, p_plus = state.x_plus, state.p_plus

            # Select state among valid states

            pass

        # Accept proposal

        # Sample momentum and simulate trajectory
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
        x[acceptance_mask] = x_prime[acceptance_mask].clone()
        x = x.detach().clone()

        if update:
            self._update(acceptance_mask)

        self.increment_n_steps()
        self.increment_n_attempted_transitions(n_chains=x.shape[0])
        self.increment_n_accepted_transitions(
            int(acceptance_mask.long().sum()))

        return x
