import pytest
import torch

from nfmc.algorithms.jump.samplers import JumpHMC, JumpMALA, JumpRWMH
from nfmc.algorithms.jump.kernels import JumpHMCKernel, JumpMALAKernel, JumpRWMHKernel
from nfmc.algorithms.preconditioning.base import PreconditionedMCMCSampler
from nfmc.algorithms.util.samples import Samples
from nfmc.util import create_flow_object
from test.util import StandardGaussian


@pytest.mark.parametrize('event_shape', [(2,), (10,), (2, 3, 5)])
@pytest.mark.parametrize('kernel_class', [JumpRWMHKernel, JumpHMCKernel, JumpMALAKernel])
@pytest.mark.parametrize('global_kernel', ['imh', 'i-sir'])
@pytest.mark.parametrize('n_chains', [1, 2, 4])
def test_jump_mh_kernel_step(event_shape,
                             kernel_class,
                             global_kernel,
                             n_chains):
    torch.manual_seed(0)

    kernel = kernel_class(
        event_shape=event_shape,
        neg_log_prob_target=StandardGaussian(event_shape).neg_log_prob,
        global_kernel=global_kernel
    )

    z_current = torch.randn(size=(n_chains, *event_shape))
    z_new = kernel.step(z_current)

    assert not z_new.requires_grad
    assert z_new.shape == z_current.shape
    assert torch.isfinite(z_new).all()
    assert z_current.dtype == z_new.dtype


@pytest.mark.parametrize('event_shape', [(2,)])
@pytest.mark.parametrize('sampler_class', [JumpRWMH, JumpHMC, JumpMALA])
@pytest.mark.parametrize('global_kernel', ['imh', 'i-sir'])
@pytest.mark.parametrize('n_chains', [4])
@pytest.mark.parametrize('n_steps', [4, 5, 6])
def test_jump_mh_warmup(event_shape,
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
    assert sampler.preconditioner is not None
    assert sampler.preconditioner.inverse_transform is not None
    assert sampler.kernel.neg_log_prob_target is not original_neg_log_prob_target

    z_initial = torch.randn(size=(n_chains, *event_shape))
    samples = sampler.warmup(
        z_initial,
        n_steps=n_steps,
        show_progress=False,
        preconditioner_update_interval=5,
        n_epochs=2  # Number of NF training epochs
    )

    assert isinstance(samples, Samples)
    assert torch.isfinite(samples.as_tensor()).all()
    assert samples.as_tensor().shape == (n_steps, n_chains, *event_shape)
    assert samples.as_tensor().dtype == z_initial.dtype


@pytest.mark.parametrize('event_shape', [(2,)])
@pytest.mark.parametrize('sampler_class', [JumpRWMH, JumpHMC, JumpMALA])
@pytest.mark.parametrize('global_kernel', ['imh', 'i-sir'])
@pytest.mark.parametrize('n_chains', [1, 2, 4])
@pytest.mark.parametrize('n_steps', [1, 2, 4])
def test_jump_mh_sample(event_shape,
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
    assert sampler.preconditioner is not None
    assert sampler.preconditioner.inverse_transform is not None
    assert sampler.kernel.neg_log_prob_target is not original_neg_log_prob_target

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
