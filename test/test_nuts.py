import torch
from nfmc.algorithms.nuts import NUTSKernel, _build_tree, _kinetic_energy, ParallelTreeState
from nfmc.util import sum_except_batch
import pytest

from test.util import StandardGaussian


@pytest.mark.parametrize('n_chains', [1, 4, 20])
@pytest.mark.parametrize('tree_depth', [0, 1, 4])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_build_tree(n_chains, tree_depth, dtype):
    torch.manual_seed(0)

    event_shape = (4,)

    # n_chains = 20
    # tree_depth: int = 3

    def neg_log_prob_target(_tensor):
        return sum_except_batch(_tensor ** 2, event_shape)

    x0 = torch.randn(size=(n_chains, *event_shape), dtype=dtype)
    p0 = torch.randn(size=(n_chains, *event_shape), dtype=dtype)
    log_prob_x0 = -neg_log_prob_target(x0).to(x0)
    joint0 = log_prob_x0 - _kinetic_energy(p0, event_shape)
    u_slice = torch.rand(size=(n_chains,), dtype=dtype) * torch.exp(joint0)

    # {0, 1} -> {0, 2} -> {-1, 1}
    v = torch.randint(low=0, high=2, size=(n_chains,)) * 2 - 1
    step_size = torch.rand(size=(n_chains,), dtype=dtype) / 100
    j = torch.full(size=(n_chains,), fill_value=tree_depth)

    state, _, _ = _build_tree(
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
    state: ParallelTreeState

    assert isinstance(state, ParallelTreeState)
    assert state.n_chains == n_chains
    assert state.event_shape == event_shape
    assert state.x_minus.shape == (n_chains, *event_shape)
    assert state.x_plus.shape == (n_chains, *event_shape)
    assert state.p_minus.shape == (n_chains, *event_shape)
    assert state.p_plus.shape == (n_chains, *event_shape)

    assert state.alpha_prime.shape == (n_chains,)
    assert state.diverged.shape == (n_chains,)
    assert state.n_alpha_prime.shape == (n_chains,)

    assert torch.isfinite(state.x_minus).all()
    assert torch.isfinite(state.x_plus).all()
    assert torch.isfinite(state.p_minus).all()
    assert torch.isfinite(state.p_plus).all()
    assert torch.isfinite(state.x_prime).all()
    assert torch.isfinite(state.log_prob_prime).all()

    assert torch.isfinite(state.alpha_prime).all()
    assert torch.isfinite(state.diverged).all()
    assert torch.isfinite(state.n_alpha_prime).all()

    assert torch.all(state.x_prime != x0)


@pytest.mark.parametrize('event_shape', [(1,), (4,), (2, 3)])
@pytest.mark.parametrize('n_chains', [1, 4])
def test_overwrite_with(event_shape, n_chains):
    torch.manual_seed(0)
    state = ParallelTreeState(
        n_chains=n_chains,
        event_shape=event_shape
    )
    left = ParallelTreeState(
        n_chains=1,
        event_shape=event_shape,
        x_minus=torch.randn(size=(1, *event_shape))
    )
    mask = torch.tensor([True] + [False] * (n_chains - 1))
    state.overwrite_with(left, mask)
    assert torch.all(state.x_minus[0] == left.x_minus[0])


@pytest.mark.parametrize('event_shape', [(1,), (4,), (2, 3)])
@pytest.mark.parametrize('n_chains', [4])
@pytest.mark.parametrize('float_dtype', [torch.float32, torch.float64])
def test_masked_copy_dtype(event_shape, n_chains, float_dtype):
    state: ParallelTreeState = ParallelTreeState(
        n_chains=n_chains,
        event_shape=event_shape
    )
    state.x_minus = state.x_minus.to(float_dtype)
    state.x_plus = state.x_plus.to(float_dtype)
    state.p_minus = state.p_minus.to(float_dtype)
    state.p_plus = state.p_plus.to(float_dtype)
    state.x_prime = state.x_prime.to(float_dtype)
    state.log_prob_prime = state.log_prob_prime.to(float_dtype)

    mask = torch.tensor([False, False, True, True], dtype=torch.bool)
    state_copy: ParallelTreeState = state.masked_copy(mask)

    assert state.x_minus.dtype == state_copy.x_minus.dtype
    assert state.x_plus.dtype == state_copy.x_plus.dtype

    assert state.p_minus.dtype == state_copy.p_minus.dtype
    assert state.p_plus.dtype == state_copy.p_plus.dtype

    assert state.x_prime.dtype == state_copy.x_prime.dtype
    assert state.log_prob_prime.dtype == state_copy.log_prob_prime.dtype

    assert state.alpha_prime.dtype == state_copy.alpha_prime.dtype
    assert state.diverged.dtype == state_copy.diverged.dtype
    assert state.n_alpha_prime.dtype == state_copy.n_alpha_prime.dtype


@pytest.mark.parametrize('event_shape', [(1,), (4,), (2, 3)])
@pytest.mark.parametrize('n_chains', [1, 4])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_step(event_shape, n_chains, dtype):
    torch.manual_seed(0)

    x_current = torch.randn(size=(n_chains, *event_shape), dtype=dtype)
    kernel = NUTSKernel(
        event_shape=event_shape,
        neg_log_prob_target=StandardGaussian(event_shape).neg_log_prob,
        max_tree_depth=5
    )
    x_new = kernel.step(x_current)

    assert x_new is not x_current
    assert not x_new.requires_grad
    assert x_new.shape == x_current.shape
    assert torch.isfinite(x_new).all()
    assert x_current.dtype == x_new.dtype
    assert torch.all(x_current != x_new)
