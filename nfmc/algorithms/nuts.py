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
    s_prime: torch.Tensor = None
    n_prime: torch.Tensor = None

    alpha_prime: torch.Tensor = None
    diverged: torch.Tensor = None
    n_alpha_prime: torch.Tensor = None

    def __post_init__(self):
        # Position
        if self.x_minus is None:
            self.x_minus = torch.full(
                size=(self.n_chains, *self.event_shape),
                fill_value=torch.nan,
            )

        if self.x_plus is None:
            self.x_plus = torch.full(
                size=(self.n_chains, *self.event_shape),
                fill_value=torch.nan,
            )

        # Momentum
        if self.p_minus is None:
            self.p_minus = torch.full(
                size=(self.n_chains, *self.event_shape),
                fill_value=torch.nan,
            )

        if self.p_plus is None:
            self.p_plus = torch.full(
                size=(self.n_chains, *self.event_shape),
                fill_value=torch.nan,
            )

        # Proposed state
        if self.x_prime is None:
            self.x_prime = torch.full(
                size=(self.n_chains, *self.event_shape),
                fill_value=torch.nan,
            )

        if self.log_prob_prime is None:
            self.log_prob_prime = torch.full(
                size=(self.n_chains,),
                fill_value=torch.nan,
            )

        # Other
        if self.s_prime is None:
            self.s_prime = torch.zeros(size=(self.n_chains,), dtype=torch.bool)

        if self.n_prime is None:
            self.n_prime = torch.zeros(size=(self.n_chains,), dtype=torch.long)

        if self.alpha_prime is None:
            self.alpha_prime = torch.zeros(size=(self.n_chains,))

        if self.diverged is None:
            self.diverged = torch.zeros(
                size=(self.n_chains,),
                dtype=torch.bool
            )

        if self.n_alpha_prime is None:
            self.n_alpha_prime = torch.zeros(
                size=(self.n_chains,),
                dtype=torch.long
            )

    def masked_copy(self, mask: torch.Tensor):
        """
        Create a copy of this state, retaining only masked chains.
        """
        return ParallelTreeState(
            n_chains=int(mask.sum()),
            event_shape=self.event_shape,
            x_minus=self.x_minus[mask],
            x_plus=self.x_plus[mask],

            p_minus=self.p_minus[mask],
            p_plus=self.p_plus[mask],

            x_prime=self.x_prime[mask],
            log_prob_prime=self.log_prob_prime[mask],

            s_prime=self.s_prime[mask],
            n_prime=self.n_prime[mask],

            alpha_prime=self.alpha_prime[mask],
            diverged=self.diverged[mask],
            n_alpha_prime=self.n_alpha_prime[mask],
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
        other_state: ParallelTreeState

        self.x_minus[mask] = other_state.x_minus
        self.x_plus[mask] = other_state.x_plus

        self.p_minus[mask] = other_state.p_minus
        self.p_plus[mask] = other_state.p_plus

        self.x_prime[mask] = other_state.x_prime
        self.log_prob_prime[mask] = other_state.log_prob_prime

        # True: trajecory should continue
        self.s_prime[mask] = other_state.s_prime
        self.n_prime[mask] = other_state.n_prime

        self.alpha_prime[mask] = other_state.alpha_prime
        self.diverged[mask] = other_state.diverged
        self.n_alpha_prime[mask] = other_state.n_alpha_prime


def _acceptance_prob(log_prob_new,
                     momentum_new,
                     log_prob_old,
                     momentum_old,
                     event_shape):
    h_new = -log_prob_new + _kinetic_energy(momentum_new, event_shape)
    h_old = -log_prob_old + _kinetic_energy(momentum_old, event_shape)
    return torch.clamp(torch.exp(h_old - h_new), max=1.0)


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
                log_u_slice: torch.Tensor,
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
    :param torch.Tensor log_u_slice: log slice variable tensor with shape `(n_chains,)`.
    :param torch.Tensor v: direction tensor with shape `(n_chains,)`.
    :param torch.Tensor j: tree depth tensor with shape `(n_chains,)`.
    :param torch.Tensor step_size: step size tensor with shape `(n_chains,)`.
    :param callable neg_log_prob_target: negative log probability density function. 
        Takes as input an event tensor with shape `(n_chains, *event_shape)` and outputs 
        a negative log probability tensor with shape `(n_chains,)`.
    :param torch.Tensor log_prob_x: log probability density of incoming position tensor x with shape `(n_chains,)`.
    :param float max_delta: energy error divergence threshold.
    """
    nc = 0
    ng = 0
    n_chains = x.shape[0]

    # Create a new state
    state = ParallelTreeState(
        n_chains=n_chains,
        event_shape=event_shape
    )
    state.x_minus = state.x_minus.to(x.dtype)
    state.x_plus = state.x_plus.to(x.dtype)
    state.p_minus = state.p_minus.to(x.dtype)
    state.p_plus = state.p_plus.to(x.dtype)
    state.x_prime = state.x_prime.to(x.dtype)
    state.log_prob_prime = state.log_prob_prime.to(x.dtype)

    if v.shape != (n_chains,):
        raise ValueError(
            f"Expected v.shape = ({n_chains},), but got {v.shape=}")

    m_b = (j == 0)  # Base case mask
    m_g = (j > 0)   # General case mask

    if torch.any(j < 0):
        raise ValueError("Incorrect tree depth")
    if torch.numel(j) == 0:
        raise ValueError("Zero chains in recursive call")

    # Base case
    if m_b.any():
        # Perform a single leapfrog step
        x1, p1, nc_b, ng_b, neg_log_prob1, _ = leapfrog_step(
            x=x[m_b].clone(),
            momentum=p[m_b].clone(),
            event_shape=event_shape,
            step_size=v[m_b] * step_size[m_b],
            neg_log_prob_target=neg_log_prob_target,
            return_neg_log_prob_and_grad=True
        )
        log_prob1 = -neg_log_prob1
        joint = log_prob1 - _kinetic_energy(p1, event_shape)

        # Set position and momentum
        state.x_minus[m_b], state.p_minus[m_b] = x1, p1
        state.x_plus[m_b], state.p_plus[m_b] = x1, p1

        # Set proposed state
        state.x_prime[m_b], state.log_prob_prime[m_b] = x1, log_prob1

        # Set divergence variables
        state.n_prime[m_b] = (log_u_slice[m_b] <= joint).long()
        state.s_prime[m_b] = (log_u_slice[m_b] - max_delta < joint)
        state.diverged[m_b] = ~torch.isfinite(joint)

        # Set other
        state.alpha_prime[m_b] = _acceptance_prob(
            log_prob1,
            p1,
            log_prob_x[m_b],
            p[m_b],
            event_shape
        ).to(state.alpha_prime.dtype)
        state.n_alpha_prime[m_b] = 1

        nc += nc_b
        ng += ng_b

    # General case
    if m_g.any():
        # Build the left subtree (`n_g` chains)
        left, nc_left, ng_left = _build_tree(
            x=x[m_g],
            p=p[m_g],
            event_shape=event_shape,
            log_u_slice=log_u_slice[m_g],
            v=v[m_g],
            j=j[m_g] - 1,
            step_size=step_size[m_g],
            neg_log_prob_target=neg_log_prob_target,
            log_prob_x=log_prob_x[m_g],
            max_delta=max_delta
        )
        nc += nc_left
        ng += ng_left

        # Split the left subtree according to early stopping:
        # - left_early (`n_g_e` chains) is the tree with early stopping. For
        #       chains where _  build_tree diverged or was told to stop, do not change
        #       their subtree anymore. In the single-chain version, this would mean
        #       returning the left subtree as is.
        # - left_continue (`n_g_c` chains)
        left_early = left.masked_copy(~left.s_prime)
        m_g_early = m_g.clone()  # Mask for early-stopped chains
        m_g_early[m_g] = ~left.s_prime
        # Overwrite the early-stopped chains in the main state
        state.overwrite_with(left_early, m_g_early)
        m_g_non_early = ~m_g_early  # Mask for non-early-stopped chains

        if m_g_non_early.any():
            left_continue = left.masked_copy(m_g_non_early)

            # (`n_g_c` chains)
            m_g_negative = (v[m_g] == -1)
            m_g_positive = ~m_g_negative

            x_start, p_start = x.clone(), p.clone()  # (`n` chains)
            (
                x_start[m_g_non_early & m_g_negative],
                p_start[m_g_non_early & m_g_negative]
            ) = (
                left_continue.x_minus[m_g_negative[m_g_non_early]],
                left_continue.p_minus[m_g_negative[m_g_non_early]]
            )
            (
                x_start[m_g_non_early & m_g_positive],
                p_start[m_g_non_early & m_g_positive]
            ) = (
                left_continue.x_plus[m_g_positive[m_g_non_early]],
                left_continue.p_plus[m_g_positive[m_g_non_early]]
            )

            right, nc_right, ng_right = _build_tree(
                x=x_start[m_g_non_early],
                p=p_start[m_g_non_early],
                event_shape=event_shape,
                log_u_slice=log_u_slice[m_g_non_early],
                v=v[m_g_non_early],
                j=j[m_g_non_early] - 1,
                step_size=step_size[m_g_non_early],
                neg_log_prob_target=neg_log_prob_target,
                log_prob_x=log_prob_x[m_g_non_early],
                max_delta=max_delta
            )

            # Combine
            # > Left (non early stopped)
            state.x_minus[m_g_non_early &
                          m_g_negative] = left_continue.x_minus[m_g_negative[m_g_non_early]]
            state.p_minus[m_g_non_early &
                          m_g_negative] = left_continue.p_minus[m_g_negative[m_g_non_early]]
            state.x_plus[m_g_non_early &
                         m_g_negative] = left_continue.x_plus[m_g_negative[m_g_non_early]]
            state.p_plus[m_g_non_early &
                         m_g_negative] = left_continue.p_plus[m_g_negative[m_g_non_early]]

            # Right
            state.x_minus[m_g_non_early &
                          m_g_positive] = right.x_minus[m_g_positive[m_g_non_early]]
            state.p_minus[m_g_non_early &
                          m_g_positive] = right.p_minus[m_g_positive[m_g_non_early]]
            state.x_plus[m_g_non_early &
                         m_g_positive] = right.x_plus[m_g_positive[m_g_non_early]]
            state.p_plus[m_g_non_early &
                         m_g_positive] = right.p_plus[m_g_positive[m_g_non_early]]

            # Set proposed position
            # If1
            n_non_early_chains = int(m_g_non_early.long().sum())
            _thresh = (right.n_prime.float()) / \
                (left_continue.n_prime.float() + right.n_prime.float())
            _rand = torch.rand((n_non_early_chains,), dtype=x.dtype)
            rand_mask = _rand < _thresh

            set_idx_dst_pos = torch.arange(n_chains)
            set_idx_dst_pos = set_idx_dst_pos[m_g_non_early]
            set_idx_dst_pos = set_idx_dst_pos[rand_mask]

            set_idx_dst_neg = torch.arange(n_chains)
            set_idx_dst_neg = set_idx_dst_neg[m_g_non_early]
            set_idx_dst_neg = set_idx_dst_neg[~rand_mask]

            # If2
            (
                state.x_prime[set_idx_dst_pos],
                state.log_prob_prime[set_idx_dst_pos]
            ) = (
                right.x_prime[rand_mask],
                right.log_prob_prime[rand_mask]
            )
            # Else2
            (
                state.x_prime[set_idx_dst_neg],
                state.log_prob_prime[set_idx_dst_neg]
            ) = (
                left_continue.x_prime[(~rand_mask)],
                left_continue.log_prob_prime[(~rand_mask)]
            )
            # Endif2

            state.s_prime[m_g_non_early] = (
                right.s_prime
                & (~is_uturn(
                    state.x_minus[m_g_non_early],
                    state.x_plus[m_g_non_early],
                    state.p_minus[m_g_non_early],
                    state.p_plus[m_g_non_early],
                    event_shape=event_shape
                ))
            )
            state.n_prime[m_g_non_early] = left_continue.n_prime + \
                right.n_prime

            state.alpha_prime[m_g_non_early] = (
                left_continue.alpha_prime
                + right.alpha_prime
            )
            state.n_alpha_prime[m_g_non_early] = (
                left_continue.n_alpha_prime
                + right.n_alpha_prime
            )
            state.diverged[m_g_non_early] = left_continue.diverged | right.diverged

            nc += nc_right
            ng += ng_right

    return state, nc, ng


class NUTSKernel(LocalMHKernel):
    """
    Implementation of the no-U-turn sampler (NUTS) transition kernel.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 neg_log_prob_target: callable,
                 max_tree_depth: int = 5,
                 max_delta: float = 1000.0,
                 target_acceptance_rate: float = 0.8,
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
        super().__init__(
            event_shape, 
            neg_log_prob_target, 
            target_acceptance_rate=target_acceptance_rate,
            **kwargs
        )
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
        x = x.clone()
        n_chains = x.shape[0]

        # Choose step size according to warmup flag
        if self.warmup_active:
            step_size = self._dual_averaging.value
        else:
            step_size = self.step_size

        # Ensure step size has appropriate shape
        if step_size.shape != (n_chains,):
            if step_size.shape != ():
                raise ValueError(
                    f"Step size should have `{n_chains=}` elements or a single one, but got {step_size.shape=}"
                )
            step_size = torch.full(
                size=(n_chains,),
                fill_value=step_size.item()
            )
        step_size = step_size.to(x.dtype)

        log_prob_x = -self.neg_log_prob_target(x).to(x.dtype)

        # Sample momentum and slice variable
        p0 = torch.randn_like(x)
        joint0 = log_prob_x - _kinetic_energy(p0, self.event_shape)
        log_u_slice = torch.log(torch.rand_like(step_size)) + joint0

        # Initialize balanced binary tree
        x_minus, x_plus = x.clone(), x.clone()
        p_minus, p_plus = p0.clone(), p0.clone()
        x_prime = x.clone()
        log_prob_x_prime = log_prob_x.clone()

        # Initialize other variables
        stop = torch.zeros(size=(n_chains,), dtype=torch.bool)  # ~s_prime
        n = torch.ones(size=(n_chains,), dtype=torch.long)

        alpha = torch.zeros(size=(n_chains,))
        n_alpha = torch.zeros(size=(n_chains,), dtype=torch.long)

        for j in range(self.max_tree_depth + 1):
            _active_mask = ~stop
            _n_active = int(torch.sum(_active_mask.long()))

            # Choose direction
            _r = torch.randint(size=(_n_active,), low=0,
                               high=2).to(step_size.dtype)
            v = _r * 2 - 1

            # Build tree into negative/positive time directions
            # (determined according to v within _build_tree)
            m_neg = (v == -1)

            _x_build_tree = x_plus[_active_mask]
            _x_build_tree[m_neg] = x_minus[_active_mask][m_neg]
            _p_build_tree = p_plus[_active_mask]
            _p_build_tree[m_neg] = p_minus[_active_mask][m_neg]

            half_tree, nc, ng = _build_tree(
                x=_x_build_tree,
                p=_p_build_tree,
                event_shape=self.event_shape,
                log_u_slice=log_u_slice[_active_mask],
                v=v,
                j=torch.full(size=(_n_active,),
                             fill_value=j, dtype=torch.long),
                step_size=step_size[_active_mask],
                neg_log_prob_target=self.neg_log_prob_target,
                log_prob_x=log_prob_x[_active_mask],
                max_delta=self.max_delta
            )
            half_tree: ParallelTreeState
            self.increment_n_divergences(int(half_tree.diverged.long().sum()))

            set_idx_dst_pos = torch.arange(n_chains)
            set_idx_dst_pos = set_idx_dst_pos[_active_mask]
            set_idx_dst_pos = set_idx_dst_pos[~m_neg]

            set_idx_dst_neg = torch.arange(n_chains)
            set_idx_dst_neg = set_idx_dst_neg[_active_mask]
            set_idx_dst_neg = set_idx_dst_neg[m_neg]

            (
                x_minus[set_idx_dst_neg],
                p_minus[set_idx_dst_neg]
            ) = (
                half_tree.x_minus[m_neg],
                half_tree.p_minus[m_neg]
            )

            (
                x_plus[set_idx_dst_pos],
                p_plus[set_idx_dst_pos]
            ) = (
                half_tree.x_plus[~m_neg],
                half_tree.p_plus[~m_neg]
            )
            stop[_active_mask] |= (~half_tree.s_prime) | half_tree.diverged

            # Update state (select state among valid states)
            _rand = torch.rand_like(step_size[_active_mask])
            _thresh = half_tree.n_prime.to(
                _rand.dtype) / n[_active_mask].to(_rand.dtype)
            m_update = (_rand < _thresh)

            set_idx_dst_update = torch.arange(n_chains)
            set_idx_dst_update = set_idx_dst_update[_active_mask]
            set_idx_dst_update = set_idx_dst_update[m_update]

            x_prime[set_idx_dst_update] = half_tree.x_prime[m_update].clone()
            log_prob_x_prime[set_idx_dst_update] = half_tree.log_prob_prime[m_update].clone(
            )

            n[_active_mask] += half_tree.n_prime
            alpha[_active_mask] += half_tree.alpha_prime
            n_alpha[_active_mask] += half_tree.n_alpha_prime

            if stop.all():
                break

        # Dual averaging statistic
        if update:
            nonzero_n_alpha_mask = (n_alpha > 0)
            if nonzero_n_alpha_mask.any():
                h_da = torch.zeros_like(alpha)[nonzero_n_alpha_mask]
                _ratio = torch.clip(
                    torch.divide(
                        alpha[nonzero_n_alpha_mask],
                        n_alpha[nonzero_n_alpha_mask].to(alpha.dtype)
                    ),
                    min=0.0,
                    max=1.0
                )
                h_da = self._target_acceptance_rate - _ratio
                self._update(h_da, nonzero_n_alpha_mask)

        self.increment_n_calls(nc)
        self.increment_n_grads(ng)
        self.increment_n_steps()
        self.increment_n_attempted_transitions(n_chains=n_chains)
        self.increment_n_accepted_transitions(n_chains)

        return x_prime

    def _update(self, h: torch.Tensor, update_mask: torch.Tensor):
        """
        Update kernel parameters.

        :param torch.Tensor h: statistic tensor after kernel transition.
        """
        if not torch.isfinite(self.step_size).all():
            raise ValueError("Step size is NaN or Inf")

        if not torch.isfinite(h).all():
            raise ValueError("Dual averaging statistic is NaN or Inf")

        self._dual_averaging.step(h, update_mask)

        # Clip step size to avoid divergences during warmup
        self.step_size = torch.clip(
            torch.mean(self._dual_averaging.value),
            min=1e-6,
            max=1e+1
        )
        if not torch.isfinite(self.step_size).all():
            raise ValueError("Step size is NaN or Inf (after update)")

    def pbar_repr(self, elapsed_time_seconds: float):
        if self._dual_averaging is not None:
            eps_mean = torch.mean(torch.as_tensor(self._dual_averaging.error_sum))
            eps_max = torch.max(torch.as_tensor(self._dual_averaging.error_sum))
            eps_min = torch.min(torch.as_tensor(self._dual_averaging.error_sum))
            da_str = f'DA[{eps_mean:.2f} ^{eps_max:.2f} v{eps_min:.2f}]'
        else:
            da_str = 'DA[None]'
        data = [
            self.name,
            f'log step: {torch.log(self.step_size):.3f}',
            da_str,
            f'{self.calls_per_second(elapsed_time_seconds):.3f} c/s',
            f'{self.grads_per_second(elapsed_time_seconds):.3f} g/s',
            f'{self.acceptance_rate:.3f} acc',
            f'{self._n_divergences} divs'
        ]
        return ', '.join(data)
