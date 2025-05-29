import pytest
import torch

from nfmc.algorithms.iterated_sir import IteratedSIRKernel
from nfmc.algorithms.jump.samplers import NeuTraJumpHMC, NeuTraJumpMALA, NeuTraJumpRWMH
from nfmc.algorithms.mh.imh import IMHKernel
from nfmc.algorithms.mh.local.hmc import HMCKernel
from nfmc.algorithms.mh.local.mala import MALAKernel
from nfmc.algorithms.mh.local.rwmh import RWMHKernel
from nfmc.algorithms.preconditioning.preconditioners import DenseLinearPreconditioner, DiagonalLinearPreconditioner, NormalizingFlowPreconditioner
from nfmc.algorithms.preconditioning.samplers.dense import DenseHMC, DenseMALA, DenseRWMH
from nfmc.algorithms.preconditioning.samplers.diagonal import DiagonalHMC, DiagonalMALA, DiagonalRWMH
from nfmc.algorithms.preconditioning.samplers.neutra import NeuTraRWMH, NeuTraMALA, NeuTraHMC
from nfmc.algorithms.preconditioning.samplers.base import PreconditionedMCMCSampler
from nfmc.algorithms.util.samples import Samples
from nfmc.util import create_flow_object
from test.util import DiagonalGaussian, StandardGaussian
from torchflows.architectures import RealNVP
from torchflows.flows import Flow


@pytest.mark.local_only
@pytest.mark.parametrize('event_shape', [(2,)])
@pytest.mark.parametrize('sampler_class', [NeuTraRWMH, NeuTraMALA, NeuTraHMC])
@pytest.mark.parametrize('n_chains', [4])
@pytest.mark.parametrize('n_steps', [4, 5, 6])
def test_warmup(event_shape,
                sampler_class,
                n_chains,
                n_steps):
    torch.manual_seed(0)
    original_neg_log_prob_target = StandardGaussian(event_shape).neg_log_prob

    flow = create_flow_object('realnvp', event_shape)
    sampler = sampler_class(
        flow=flow,
        neg_log_prob_target=original_neg_log_prob_target,
    )
    assert sampler.kernel._preconditioner is not None
    assert sampler.kernel._preconditioner.inverse_transform is not None
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


@pytest.mark.local_only
@pytest.mark.parametrize('event_shape', [(2,), (2, 3, 5)])
@pytest.mark.parametrize('sampler_class', [
    DiagonalRWMH,
    DiagonalMALA,
    DiagonalHMC,
    DenseRWMH,
    DenseMALA,
    DenseHMC,
    NeuTraRWMH,
    NeuTraMALA,
    NeuTraHMC,
])
@pytest.mark.parametrize('n_chains', [1, 4])
@pytest.mark.parametrize('n_steps', [1, 4])
def test_sample_shape(event_shape,
                      sampler_class,
                      n_chains,
                      n_steps):
    torch.manual_seed(0)
    original_neg_log_prob_target = StandardGaussian(event_shape).neg_log_prob

    if sampler_class in [NeuTraHMC, NeuTraRWMH, NeuTraMALA]:
        flow = create_flow_object('realnvp', event_shape)
        sampler = sampler_class(
            flow=flow,
            neg_log_prob_target=original_neg_log_prob_target,
        )
    else:
        sampler = sampler_class(
            event_shape,
            original_neg_log_prob_target
        )

    assert sampler.kernel._preconditioner is not None
    assert sampler.kernel._preconditioner.inverse_transform is not None
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


@pytest.mark.parametrize('event_shape', [(2,), (10,), (2, 3, 5)])
@pytest.mark.parametrize('kernel_class', [RWMHKernel, HMCKernel, MALAKernel, IMHKernel])
@pytest.mark.parametrize('n_chains', [1, 2, 4])
@pytest.mark.parametrize('n_steps', [1, 2, 4])
def test_sample_nf(event_shape, kernel_class, n_chains, n_steps):
    torch.manual_seed(0)

    flow = create_flow_object('realnvp', event_shape)

    kernel = kernel_class(
        event_shape=event_shape,
        neg_log_prob_target=StandardGaussian(event_shape).neg_log_prob,
        preconditioner=NormalizingFlowPreconditioner(flow)
    )
    sampler = PreconditionedMCMCSampler(kernel)

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
        NeuTraRWMH,
        NeuTraMALA,
        NeuTraHMC,
        DiagonalRWMH,
        DiagonalMALA,
        DiagonalHMC,
        DenseRWMH,
        DenseMALA,
        DenseHMC,
    ]
)
def test_local_mh_warmup_and_sample(sampler_class):
    torch.manual_seed(0)

    event_shape = (2,)
    n_chains = 50
    target = DiagonalGaussian(event_shape, mu=1.5, std=0.5)

    if sampler_class in [NeuTraHMC, NeuTraMALA, NeuTraRWMH]:
        flow = Flow(RealNVP(event_shape, n_layers=1))
        sampler = sampler_class(
            flow=flow,
            neg_log_prob_target=target.neg_log_prob
        )
    else:
        sampler = sampler_class(
            event_shape=event_shape,
            neg_log_prob_target=target.neg_log_prob
        )

    z0 = torch.rand(size=(n_chains, *event_shape)) * 2 - 1
    _, latent_warmup_draws = sampler.warmup(
        z0=z0,
        n_steps=500,
        preconditioner_update_interval=100,
        return_latent_samples=True,
    )
    sampling_draws = sampler.sample(
        z0=latent_warmup_draws.last_sample,
        n_steps=400,
    )

    if sampler_class in [NeuTraHMC, NeuTraMALA, NeuTraRWMH]:
        # Check flow validity for NF preconditioning
        flow_samples = flow.sample((10000,)).detach()
        flow_first_moment = flow_samples.mean(0)
        flow_second_moment = flow_samples.square().mean(0)
        assert torch.allclose(target.first_moment, flow_first_moment, rtol=0.2)
        assert torch.allclose(target.second_moment,
                              flow_second_moment, rtol=0.2)

    # Check MCMC sample validity
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


@pytest.mark.local_only
@pytest.mark.parametrize('preconditioner', ['diag', 'dense', 'nf'])
def test_imh(preconditioner):
    torch.manual_seed(0)

    event_shape = (4,)
    n_chains = 50
    target = DiagonalGaussian(event_shape, mu=1.5, std=0.5)

    if preconditioner == 'nf':
        preconditioner = NormalizingFlowPreconditioner(
            Flow(RealNVP(event_shape, n_layers=1))
        )
    elif preconditioner == 'diag':
        preconditioner = DiagonalLinearPreconditioner(event_shape)
    elif preconditioner == 'dense':
        preconditioner = DenseLinearPreconditioner(event_shape)
    else:
        raise ValueError

    torch.manual_seed(0)
    kernel = IMHKernel(
        event_shape,
        target.neg_log_prob,
        preconditioner=preconditioner
    )
    sampler = PreconditionedMCMCSampler(kernel)

    z0 = torch.rand(size=(n_chains, *event_shape)) * 2 - 1

    _, latent_warmup_draws = sampler.warmup(
        z0,
        n_steps=200,
        preconditioner_update_interval=50,
        return_latent_samples=True
    )
    sampling_draws = sampler.sample(
        latent_warmup_draws.last_sample,
        n_steps=4000
    )

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


@pytest.mark.local_only
@pytest.mark.parametrize('preconditioner', ['diag', 'dense', 'nf'])
def test_isir(preconditioner):
    torch.manual_seed(0)

    event_shape = (4,)
    n_chains = 50
    target = DiagonalGaussian(event_shape, mu=1.5, std=0.5)

    if preconditioner == 'nf':
        preconditioner = NormalizingFlowPreconditioner(
            Flow(RealNVP(event_shape, n_layers=1))
        )
    elif preconditioner == 'diag':
        preconditioner = DiagonalLinearPreconditioner(event_shape)
    elif preconditioner == 'dense':
        preconditioner = DenseLinearPreconditioner(event_shape)
    else:
        raise ValueError

    torch.manual_seed(0)
    kernel = IteratedSIRKernel(
        event_shape,
        target.neg_log_prob,
        preconditioner=preconditioner
    )
    sampler = PreconditionedMCMCSampler(kernel)

    z0 = torch.rand(size=(n_chains, *event_shape)) * 2 - 1
    _, latent_warmup_draws = sampler.warmup(
        z0,
        n_steps=200,
        preconditioner_update_interval=50,
        return_latent_samples=True
    )
    sampling_draws = sampler.sample(
        latent_warmup_draws.last_sample,
        n_steps=400
    )

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


@pytest.mark.local_only
@pytest.mark.parametrize(
    "sampler_class", [
        NeuTraJumpRWMH,
        NeuTraJumpMALA,
        NeuTraJumpHMC,
    ]
)
@pytest.mark.parametrize(
    "global_kernel", [
        'imh',
        'i-sir'
    ]
)
def test_jump_mcmc_warmup_and_sample(sampler_class, global_kernel):
    torch.manual_seed(0)

    event_shape = (4,)
    n_chains = 50
    target = DiagonalGaussian(event_shape, mu=1.5, std=0.5)
    flow = Flow(RealNVP(event_shape))

    sampler = sampler_class(
        flow=flow,
        neg_log_prob_target=target.neg_log_prob,
        global_kernel=global_kernel
    )

    z0 = torch.rand(size=(n_chains, *event_shape)) * 2 - 1
    _, latent_warmup_draws = sampler.warmup(
        z0,
        n_steps=400,
        preconditioner_update_interval=50,
        return_latent_samples=True
    )
    sampling_draws = sampler.sample(
        latent_warmup_draws.last_sample,
        n_steps=400
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
