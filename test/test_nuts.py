import torch
from nfmc.algorithms.nuts import _build_tree, _kinetic_energy, ParallelTreeState
from nfmc.util import sum_except_batch
import pytest

@pytest.mark.parametrize('n_chains', [1, 4, 20])
@pytest.mark.parametrize('tree_depth', [0, 1, 4, 10])
def test_build_tree(n_chains, tree_depth):
    torch.manual_seed(0)
    
    event_shape = (4,)

    # n_chains = 20
    # tree_depth: int = 3

    def neg_log_prob_target(_tensor):
        return sum_except_batch(_tensor ** 2, event_shape)
    
    x0 = torch.randn(size=(n_chains, *event_shape))
    p0 = torch.randn(size=(n_chains, *event_shape))
    log_prob_x0 = -neg_log_prob_target(x0)
    joint0 = log_prob_x0 - _kinetic_energy(p0, event_shape)
    u_slice = torch.rand(size=(n_chains,)) * torch.exp(joint0)

    # {0, 1} -> {0, 2} -> {-1, 1}
    v = torch.randint(low=0, high=2, size=(n_chains,)) * 2 - 1
    step_size = torch.rand(size=(n_chains,)) / 100
    j = torch.full(size=(n_chains,), fill_value=tree_depth)

    state: ParallelTreeState = _build_tree(
        x=x0,
        p=p0,
        event_shape=event_shape,
        u_slice=u_slice,
        v=v,
        j=j,
        step_size=step_size,
        neg_log_prob_target=neg_log_prob_target,
        log_prob_x=log_prob_x0,
        max_delta=1000.0
    )

    assert state.n_chains == n_chains
    assert state.event_shape == event_shape
    assert state.x_minus.shape == (n_chains, *event_shape)
    assert state.x_plus.shape == (n_chains, *event_shape)
    assert state.p_minus.shape == (n_chains, *event_shape)
    assert state.p_plus.shape == (n_chains, *event_shape)

    assert state.n_valid.shape == (n_chains,)
    assert state.sum_accept_prob.shape == (n_chains,)
    assert state.stop.shape == (n_chains,)
    assert state.diverged.shape == (n_chains,)
    assert state.n_leapfrogs.shape == (n_chains,)

    assert torch.isfinite(state.x_minus).all()
    assert torch.isfinite(state.x_plus).all()
    assert torch.isfinite(state.p_minus).all()
    assert torch.isfinite(state.p_plus).all()
    assert torch.isfinite(state.x_prime).all()
    assert torch.isfinite(state.log_prob_prime).all()

    assert torch.isfinite(state.n_valid).all()
    assert torch.isfinite(state.sum_accept_prob).all()
    assert torch.isfinite(state.stop).all()
    assert torch.isfinite(state.diverged).all()
    assert torch.isfinite(state.n_leapfrogs).all()

    assert torch.all(state.x_prime != x0)