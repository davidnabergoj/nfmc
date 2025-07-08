import pytest
import torch

from nfmc.algorithms.jump.samplers import DiagonalJumpHMC, DiagonalJumpMALA, DiagonalJumpRWMH
from nfmc.algorithms.jump.kernels import (
    NeuTraJumpHMCKernel,
    NeuTraJumpMALAKernel,
    NeuTraJumpRWMHKernel,
    DiagonalJumpHMCKernel,
    DiagonalJumpMALAKernel,
    DiagonalJumpRWMHKernel,
    DenseJumpHMCKernel,
    DenseJumpMALAKernel,
    DenseJumpRWMHKernel,
)
from nfmc.algorithms.util.samples import Samples
from nfmc.util import create_flow_object
from test.util import DiagonalGaussian, StandardGaussian
from torchflows import Flow, ElementwiseAffine


def create_small_flow(event_shape):
    return Flow(ElementwiseAffine(event_shape))


@pytest.mark.parametrize('event_shape', [(2,), (10,), (2, 3, 5)])
@pytest.mark.parametrize('kernel_class', [
    NeuTraJumpRWMHKernel,
    NeuTraJumpHMCKernel,
    NeuTraJumpMALAKernel,
    DiagonalJumpHMCKernel,
    DiagonalJumpMALAKernel,
    DiagonalJumpRWMHKernel,
    DenseJumpHMCKernel,
    DenseJumpMALAKernel,
    DenseJumpRWMHKernel,
])
@pytest.mark.parametrize('global_kernel', ['imh', 'i-sir'])
@pytest.mark.parametrize('n_chains', [1, 2, 4])
def test_step(event_shape,
              kernel_class,
              global_kernel,
              n_chains):
    torch.manual_seed(0)

    flow = create_small_flow(event_shape)
    kernel = kernel_class(
        flow=flow,
        neg_log_prob_target=StandardGaussian(event_shape).neg_log_prob,
        global_kernel=global_kernel
    )

    z_current = torch.randn(size=(n_chains, *event_shape))
    z_new = kernel.step(z_current)

    assert not z_new.requires_grad
    assert z_new.shape == z_current.shape
    assert torch.isfinite(z_new).all()
    assert z_current.dtype == z_new.dtype


@pytest.mark.local_only
@pytest.mark.parametrize('event_shape', [(2,)])
@pytest.mark.parametrize('sampler_class', [
    DiagonalJumpRWMH,
    DiagonalJumpHMC,
    DiagonalJumpMALA
])
@pytest.mark.parametrize('global_kernel', ['imh', 'i-sir'])
@pytest.mark.parametrize('n_chains', [4])
@pytest.mark.parametrize('cycle_length', [2, 3, 4])
def test_warmup(event_shape,
                sampler_class,
                global_kernel,
                n_chains,
                cycle_length):
    torch.manual_seed(0)
    n_cycles = 2
    original_neg_log_prob_target = StandardGaussian(event_shape).neg_log_prob

    flow = create_flow_object('realnvp', event_shape)
    sampler = sampler_class(
        flow=flow,
        neg_log_prob_target=original_neg_log_prob_target,
        global_kernel=global_kernel
    )

    for k in sampler.kernel.kernels:
        assert k._preconditioner is not None
        assert k._preconditioner.inverse_transform is not None
        assert k.neg_log_prob_target is not original_neg_log_prob_target

    z_initial = torch.randn(size=(n_chains, *event_shape))
    samples = sampler.warmup(
        z_initial,
        n_cycles=n_cycles,
        cycle_length=cycle_length,
        show_progress=False,
        n_epochs=2  # Number of NF training epochs
    )

    assert isinstance(samples, Samples)
    assert torch.isfinite(samples.as_tensor()).all()
    assert samples.as_tensor().shape == (
        cycle_length * n_cycles, n_chains, *event_shape
        )
    assert samples.as_tensor().dtype == z_initial.dtype


@pytest.mark.local_only
@pytest.mark.parametrize('event_shape', [(2,)])
@pytest.mark.parametrize('sampler_class', [
    DiagonalJumpRWMH,
    DiagonalJumpHMC,
    DiagonalJumpMALA
])
@pytest.mark.parametrize('global_kernel', ['imh', 'i-sir'])
@pytest.mark.parametrize('n_chains', [1, 2, 4])
@pytest.mark.parametrize('n_steps', [1, 2, 4])
def test_sample(event_shape,
                sampler_class,
                global_kernel,
                n_chains,
                n_steps):
    torch.manual_seed(0)
    original_neg_log_prob_target = StandardGaussian(event_shape).neg_log_prob

    flow = create_flow_object('realnvp', event_shape)
    sampler = sampler_class(
        flow=flow,
        neg_log_prob_target=original_neg_log_prob_target,
        global_kernel=global_kernel
    )
    for k in sampler.kernel.kernels:
        assert k._preconditioner is not None
        assert k._preconditioner.inverse_transform is not None
        assert k.neg_log_prob_target is not original_neg_log_prob_target

    z_initial = torch.randn(size=(n_chains, *event_shape))
    samples = sampler.sample(
        z_initial,
        n_steps=n_steps,
        show_progress=False
    )

    assert isinstance(samples, Samples)
    assert torch.isfinite(samples.as_tensor()).all()
    assert samples.as_tensor().shape == (n_steps, n_chains, *event_shape)
    assert samples.as_tensor().dtype == z_initial.dtype


@pytest.mark.local_only
@pytest.mark.parametrize(
    "sampler_class", [
        DiagonalJumpRWMH,
        DiagonalJumpMALA,
        DiagonalJumpHMC,
    ]
)
@pytest.mark.parametrize("global_kernel", ['imh', 'i-sir'])
def test_warmup_and_sample(sampler_class, global_kernel):
    torch.manual_seed(0)

    event_shape = (4,)
    n_chains = 50
    target = DiagonalGaussian(event_shape, mu=1.5, std=0.5)
    flow = create_small_flow(event_shape)

    sampler = sampler_class(
        flow=flow,
        neg_log_prob_target=target.neg_log_prob,
        global_kernel=global_kernel
    )

    z0 = torch.rand(size=(n_chains, *event_shape)) * 2 - 1
    _, latent_warmup_draws = sampler.warmup(
        z0=z0,
        n_cycles=12,
        cycle_length=50,
        return_latent_samples=True,
        n_epochs=2
    )
    sampling_draws, _ = sampler.sample(
        z0=latent_warmup_draws.last_sample,
        n_steps=400,
        return_latent_samples=True
    )

    flow_samples = flow.sample((10000,)).detach()
    flow_first_moment = flow_samples.mean(0)
    flow_second_moment = flow_samples.square().mean(0)
    assert torch.allclose(target.first_moment, flow_first_moment, rtol=0.2)
    assert torch.allclose(target.second_moment, flow_second_moment, rtol=0.2)

    assert torch.allclose(
        sampling_draws.first_moment.as_tensor(),
        target.first_moment,
        rtol=0.2
    )
    assert torch.allclose(
        sampling_draws.second_moment.as_tensor(),
        target.second_moment,
        rtol=0.2
    )
