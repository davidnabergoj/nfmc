from typing import Tuple, Union
from dataclasses import dataclass

import torch
from nfmc.algorithms.mh.local.base import LocalMHKernel
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
        self.n_leapfrogs = torch.zeros(size=(self.n_chains,), dtype=torch.long)
    
    def to(self, tensor: torch.Tensor):
        self.x_minus = self.x_minus.to(tensor)
        self.x_plus = self.x_plus.to(tensor)
        
        self.p_minus = self.p_minus.to(tensor)
        self.p_plus = self.p_plus.to(tensor)

        self.x_prime = self.x_prime.to(tensor)
        self.log_prob_prime = self.log_prob_prime.to(tensor)

        return self

    def masked_copy(self, mask: torch.Tensor):
        """
        Create a copy of this state, retaining only masked chains.
        """
        return ParallelTreeState(
            n_chains=int(mask.sum()),
            event_shape=self.event_shape,
            x_minus=self.x_minus[mask].clone().to(self.x_minus),
            x_plus=self.x_plus[mask].clone().to(self.x_plus),

            p_minus=self.p_minus[mask].clone().to(self.p_minus),
            p_plus=self.p_plus[mask].clone().to(self.p_plus),

            x_prime=self.x_prime[mask].clone().to(self.x_prime),
            log_prob_prime=self.log_prob_prime[mask].clone().to(self.log_prob_prime),

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
        self.x_minus[mask] = other_state.x_minus.to(self.x_minus)
        self.x_plus[mask] = other_state.x_plus.to(self.x_plus)

        self.p_minus[mask] = other_state.p_minus.to(self.p_minus)
        self.p_plus[mask] = other_state.p_plus.to(self.p_plus)

        self.x_prime[mask] = other_state.x_prime.to(self.x_prime)
        self.log_prob_prime[mask] = other_state.log_prob_prime.to(self.log_prob_prime)
        self.n_valid[mask] = other_state.n_valid
        self.sum_accept_prob[mask] = other_state.sum_accept_prob
        self.stop[mask] = other_state.stop
        self.diverged[mask] = other_state.diverged
        self.n_leapfrogs[mask] = other_state.n_leapfrogs


def _acceptance_prob(log_prob_new,
                     momentum_new,
                     log_prob_old,
                     momentum_old,
                     event_shape):
    h_new = -log_prob_new + _kinetic_energy(momentum_new, event_shape)
    h_old = -log_prob_old + _kinetic_energy(momentum_old, event_shape)
    return torch.clamp(torch.exp(h_old-h_new), max=1.0)


def is_uturn(x_minus,
             x_plus,
             p_minus,
             p_plus,
             event_shape):
    dx = x_plus - x_minus
    m1 = sum_except_batch(dx * p_minus, event_shape) < 0
    m2 = sum_except_batch(dx * p_plus, event_shape) < 0
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
                max_delta: float) -> Tuple[ParallelTreeState, int, int]:
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
    ).to(x)

    m_b = (j == 0)  # Base case mask
    m_g = (j > 0)   # General case mask

    if torch.any(j < 0):
        raise ValueError("Incorrect tree depth")
    if torch.numel(j) == 0:
        raise ValueError("Zero chains in recursive call")

    # Base case
    if m_b.any():
        # Perform a single leapfrog step
        x1, p1, nc, ng, neg_log_prob1, g1 = leapfrog_step(
            x=x[m_b],
            momentum=p[m_b],
            event_shape=event_shape,
            step_size=v[m_b] * step_size[m_b],
            neg_log_prob_target=neg_log_prob_target,
            return_neg_log_prob_and_grad=True
        )
        log_prob1 = -neg_log_prob1

        joint = log_prob1 - _kinetic_energy(p1, event_shape)
        diverged = ((joint - log_prob_x[m_b])
                    < -max_delta) | (~torch.isfinite(log_prob1))
        valid_b = (torch.log(u_slice[m_b]) <= joint) & (
            ~diverged) & torch.isfinite(log_prob1)

        state.x_minus[m_b], state.p_minus[m_b] = x1, p1
        state.x_plus[m_b], state.p_plus[m_b] = x1, p1
        state.x_prime[m_b], state.log_prob_prime[m_b] = x1, log_prob1

        state.n_valid[m_b][valid_b] = 1
        # Redundant, but kept for safety
        state.n_valid[m_b][~valid_b] = 0
        state.sum_accept_prob[m_b] = _acceptance_prob(
            log_prob1,
            p1,
            log_prob_x[m_b],
            p[m_b],
            event_shape
        )
        state.n_leapfrogs[m_b] = 1
        state.stop[m_b] = diverged
        state.diverged[m_b] = diverged

        return state, nc, ng

    # General case
    elif m_g.any():
        # Build the left subtree (`n_g` chains)
        left, nc_left, ng_left = _build_tree(
            x=x[m_g],
            p=p[m_g],
            event_shape=event_shape,
            u_slice=u_slice[m_g],
            v=v[m_g],
            j=j[m_g] - 1,
            step_size=step_size[m_g],
            neg_log_prob_target=neg_log_prob_target,
            log_prob_x=log_prob_x[m_g],
            max_delta=max_delta
        )

        # Split the left subtree according to early stopping:
        # - left_early (`n_g_e` chains) is the tree with early stopping. For
        #       chains where _  build_tree diverged or was told to stop, do not change
        #       their subtree anymore. In the single-chain version, this would mean
        #       returning the left subtree as is.
        # - left_continue (`n_g_c` chains)
        left_early = left.masked_copy(left.stop)
        m_g_early = m_g.clone()  # Mask for early-stopped chains
        m_g_early[m_g] = left.stop
        # Overwrite the early-stopped chains in the main state
        state.overwrite_with(left_early, m_g_early)
        m_g_non_early = ~m_g_early  # Mask for non-early-stopped chains

        if not m_g_non_early.any():
            return left, nc_left, ng_left

        left_continue = left.masked_copy(m_g_non_early)

        # (`n_g_c` chains)
        m_g_non_early_negative = (v[m_g_non_early] == -1)
        m_g_non_early_positive = ~m_g_non_early_negative

        x_start, p_start = x.clone(), p.clone()  # (`n` chains)
        (
            x_start[m_g_non_early][m_g_non_early_negative],
            p_start[m_g_non_early][m_g_non_early_negative]
        ) = (
            left_continue.x_minus[m_g_non_early_negative],
            left_continue.p_minus[m_g_non_early_negative]
        )
        (
            x_start[m_g_non_early][m_g_non_early_positive],
            p_start[m_g_non_early][m_g_non_early_positive]
        ) = (
            left_continue.x_plus[m_g_non_early_positive],
            left_continue.p_plus[m_g_non_early_positive]
        )

        right, nc_right, ng_right = _build_tree(
            x=x_start[m_g_non_early],
            p=p_start[m_g_non_early],
            event_shape=event_shape,
            u_slice=u_slice[m_g_non_early],
            v=v[m_g_non_early],
            j=j[m_g_non_early] - 1,
            step_size=step_size[m_g_non_early],
            neg_log_prob_target=neg_log_prob_target,
            log_prob_x=log_prob_x[m_g_non_early],
            max_delta=max_delta
        )

        # Combine
        # > Left (non early stopped)
        state.x_minus[m_g_non_early][m_g_non_early_negative] = left_continue.x_minus[m_g_non_early_negative]
        state.p_minus[m_g_non_early][m_g_non_early_negative] = left_continue.p_minus[m_g_non_early_negative]
        state.x_plus[m_g_non_early][m_g_non_early_negative] = left_continue.x_plus[m_g_non_early_negative]
        state.p_plus[m_g_non_early][m_g_non_early_negative] = left_continue.p_plus[m_g_non_early_negative]

        # Right
        state.x_minus[m_g_non_early][m_g_non_early_positive] = right.x_minus[m_g_non_early_positive]
        state.p_minus[m_g_non_early][m_g_non_early_positive] = right.p_minus[m_g_non_early_positive]
        state.x_plus[m_g_non_early][m_g_non_early_positive] = right.x_plus[m_g_non_early_positive]
        state.p_plus[m_g_non_early][m_g_non_early_positive] = right.p_plus[m_g_non_early_positive]

        # Choose a proposal uniformly from valid points
        state.n_valid[m_g_non_early] = left_continue.n_valid + right.n_valid
        valid_g_non_early = state.n_valid[m_g_non_early] > 0  # (< n_g_c)

        # If1
        rand_mask = torch.less(
            torch.rand(int(m_g_non_early.long().sum()),),
            right.n_valid / state.n_valid[m_g_non_early]
        )

        # If2
        (
            state.x_prime[m_g_non_early][rand_mask],
            state.log_prob_prime[m_g_non_early][rand_mask]
        ) = (
            right.x_prime[rand_mask],
            right.log_prob_prime[rand_mask]
        )
        # Else2
        (
            state.x_prime[m_g_non_early][~rand_mask],
            state.log_prob_prime[m_g_non_early][~rand_mask]
        ) = (
            left_continue.x_prime[~rand_mask],
            left_continue.log_prob_prime[~rand_mask]
        )
        # Endif2

        # Else1
        (
            state.x_prime[m_g_non_early][~valid_g_non_early],
            state.log_prob_prime[m_g_non_early][~valid_g_non_early]
        ) = (
            left_continue.x_prime[left_continue.n_valid == 0],
            left_continue.log_prob_prime[left_continue.n_valid == 0]
        )
        # EndIf1

        state.sum_accept_prob[m_g_non_early] = (
            left_continue.sum_accept_prob
            + right.sum_accept_prob
        )
        state.n_leapfrogs[m_g_non_early] = (
            left_continue.n_leapfrogs
            + right.n_leapfrogs
        )
        state.diverged[m_g_non_early] = left_continue.diverged | right.diverged

        state.stop[m_g_non_early] = right.stop | is_uturn(
            left_continue.x_minus,
            right.x_plus,
            left_continue.p_minus,
            right.p_plus,
            event_shape=event_shape
        )

        return state, nc_left + nc_right, ng_left + ng_right
    else:
        raise ValueError("Recursion did not use base or general case in any chain")

class NUTSKernel(LocalMHKernel):
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
        n_chains = x.shape[0]

        if self.warmup_active:
            step_size = self._dual_averaging.value
        else:
            step_size = self.step_size
            if step_size.shape != (n_chains,):
                if step_size.shape != ():
                    raise ValueError(
                        f"Step size should have `{n_chains = }` elements or a single one, but got {step_size.shape = }")
                step_size = torch.full(
                    size=(n_chains,), fill_value=step_size.item())

        log_prob_x = -self.neg_log_prob_target(x).to(x)

        # Sample momentum and slice variable
        p0 = torch.randn_like(x)
        joint0 = log_prob_x - _kinetic_energy(p0, self.event_shape)
        u_slice = torch.rand_like(step_size).to(x) * torch.exp(joint0).to(x)

        # Initialize balanced binary tree
        x_minus, x_plus = x.clone(), x.clone()
        p_minus, p_plus = p0.clone(), p0.clone()
        x_prime = x.clone()
        log_prob_x_prime = log_prob_x.clone()

        # Initialize other variables
        n_valid = torch.zeros(size=(n_chains,), dtype=torch.long)
        sum_accept = torch.zeros(size=(n_chains,))
        n_lf_total = torch.zeros(size=(n_chains,), dtype=torch.long)
        stop = torch.zeros(size=(n_chains,), dtype=torch.bool)
        diverged = torch.zeros(size=(n_chains,), dtype=torch.bool)

        for j in range(self.max_tree_depth + 1):
            # Choose direction
            v = torch.randint_like(step_size, low=0, high=2) * 2 - 1

            # Build tree into negative/positive time directions
            # (determined according to v within _build_tree)
            m_neg = (v == -1)

            _x_build_tree = x_plus.clone()
            _x_build_tree[m_neg] = x_minus[m_neg]
            _p_build_tree = p_plus.clone()
            _p_build_tree[m_neg] = p_minus[m_neg]

            state, nc, ng = _build_tree(
                x=_x_build_tree,
                p=_p_build_tree,
                event_shape=self.event_shape,
                u_slice=u_slice,
                v=v,
                j=torch.full(size=(n_chains,), fill_value=j, dtype=torch.long),
                step_size=step_size,
                neg_log_prob_target=self.neg_log_prob_target,
                log_prob_x=log_prob_x,
                max_delta=self.max_delta
            )
            state: ParallelTreeState

            (
                x_minus[m_neg],
                p_minus[m_neg]
            ) = (
                state.x_minus[m_neg],
                state.p_minus[m_neg]
            )

            (
                x_plus[~m_neg],
                p_plus[~m_neg]
            ) = (
                state.x_plus[~m_neg],
                state.p_plus[~m_neg]
            )
            stop[state.stop] = True
            diverged[state.diverged] = True

            # Update state (select state among valid states)
            _rand = torch.rand(size=(n_chains,)).to(x)
            _thresh = state.n_valid / (n_valid + state.n_valid)
            m_update = (state.n_valid > 0) & (_rand < _thresh)
            x_prime[m_update] = state.x_prime[m_update].clone()
            log_prob_x_prime[m_update] = state.log_prob_prime[m_update].clone()

            n_valid += state.n_valid
            sum_accept += state.sum_accept_prob
            n_lf_total += state.n_leapfrogs

            if stop.all():
                break
            if is_uturn(
                x_minus=x_minus,
                x_plus=x_plus,
                p_minus=p_minus,
                p_plus=p_plus,
                event_shape=self.event_shape,
            ).all():
                break

        # Accept proposal
        x = x_prime
        log_prob_x = log_prob_x_prime

        # Dual averaging statistic
        if update:
            h_da = sum_accept / torch.clip(n_lf_total, max=torch.tensor(1.0))
            self._update(h_da)

        self.increment_n_calls(nc)
        self.increment_n_grads(ng)
        self.increment_n_divergences(int(state.diverged.long().sum()))
        self.increment_n_steps()
        self.increment_n_attempted_transitions(n_chains=n_chains)
        self.increment_n_accepted_transitions(n_chains)

        return x

    def _update(self, h: torch.Tensor):
        """
        Update kernel parameters.

        :param torch.Tensor h: statistic tensor after kernel transition.
        """
        self._dual_averaging.step(h)
        self.step_size = torch.mean(self._dual_averaging.value)
