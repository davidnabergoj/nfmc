import pytest
import torch

from nfmc.algorithms.iterated_sir import IteratedSIRKernel
from nfmc.algorithms.jump.samplers import DiagonalJumpHMC, DiagonalJumpMALA, DiagonalJumpRWMH, NeuTraJumpHMC, NeuTraJumpMALA, NeuTraJumpRWMH
from nfmc.algorithms.mh.imh import IMHKernel
from nfmc.algorithms.preconditioning.preconditioners import DenseLinearPreconditioner, DiagonalLinearPreconditioner, NormalizingFlowPreconditioner
from torchflows.flows import Flow
from torchflows.architectures import RealNVP

from nfmc.algorithms.mh.local.hmc import HMCKernel
from nfmc.algorithms.mh.local.rwmh import RWMHKernel
from nfmc.algorithms.mh.local.mala import MALAKernel
from nfmc.algorithms.mh.base import MHSampler
from nfmc.algorithms.preconditioning.samplers.base import PreconditionedMCMCSampler
from nfmc.algorithms.preconditioning.samplers.neutra import (
    NeuTraHMC,
    NeuTraMALA,
    NeuTraRWMH,
)
from nfmc.algorithms.preconditioning.samplers.diagonal import (
    DiagonalHMC,
    DiagonalMALA,
    DiagonalRWMH,
)
from nfmc.algorithms.preconditioning.samplers.dense import (
    DenseHMC,
    DenseMALA,
    DenseRWMH,
)

from test.util import DiagonalGaussian


@pytest.mark.parametrize("kernel_class", [RWMHKernel, MALAKernel, HMCKernel])
@pytest.mark.local_only
def test_local_mh(kernel_class):
    torch.manual_seed(0)

    event_shape = (4,)
    target = DiagonalGaussian(event_shape)

    kernel = kernel_class(event_shape, neg_log_prob_target=target.neg_log_prob)
    sampler = MHSampler(kernel)

    x0 = torch.rand(size=(1, *event_shape)) * 2 - 1
    warmup_draws = sampler.warmup(
        x0=x0, n_steps=100 if kernel_class != RWMHKernel else 1000
    )
    sampling_draws = sampler.sample(
        x0=warmup_draws.last_sample, n_steps=200 if kernel_class != RWMHKernel else 2000
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
        DiagonalRWMH,
        DiagonalMALA,
        DiagonalHMC,
        DenseRWMH,
        DenseMALA,
        DenseHMC,
    ]
)
def test_linear_prec_mh_warmup_and_sample(sampler_class):
    torch.manual_seed(0)

    event_shape = (4,)
    n_chains = 10
    target = DiagonalGaussian(event_shape)

    sampler = sampler_class(
        event_shape=event_shape,
        neg_log_prob_target=target.neg_log_prob
    )

    z0 = torch.rand(size=(n_chains, *event_shape)) * 2 - 1
    warmup_draws, latent_warmup_draws = sampler.warmup(
        z0=z0,
        n_steps=200,
        preconditioner_update_interval=50,
        return_latent_samples=True,
    )
    sampling_draws, latent_sampling_draws = sampler.sample(
        z0=latent_warmup_draws.last_sample,
        n_steps=200,
        return_latent_samples=True
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
        NeuTraRWMH,
        NeuTraMALA,
        NeuTraHMC,
    ]
)
def test_neutra_mh(sampler_class):
    torch.manual_seed(0)

    event_shape = (2,)
    n_chains = 50
    target = DiagonalGaussian(event_shape, mu=1.5, std=0.5)
    flow = Flow(RealNVP(event_shape, n_layers=1))

    sampler = sampler_class(
        flow=flow,
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


@pytest.mark.local_only
def test_imh():
    torch.manual_seed(0)

    event_shape = (4,)
    n_chains = 10
    target = DiagonalGaussian(event_shape, mu=0.5, std=0.5)
    kernel = IMHKernel(event_shape, target.neg_log_prob)
    sampler = MHSampler(kernel)

    x0 = torch.rand(size=(n_chains, *event_shape)) * 2 - 1
    sampling_draws = sampler.sample(x0, n_steps=2000)

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
def test_neutra_imh(preconditioner):
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
def test_iterated_sir():
    torch.manual_seed(0)

    event_shape = (4,)
    n_chains = 10
    target = DiagonalGaussian(event_shape, mu=0.5, std=0.5)
    kernel = IteratedSIRKernel(event_shape, target.neg_log_prob)
    sampler = MHSampler(kernel)

    x0 = torch.rand(size=(n_chains, *event_shape)) * 2 - 1
    sampling_draws = sampler.sample(x0, n_steps=2000)

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
def test_neutra_iterated_sir(preconditioner):
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
def test_neutra_jump_mh(sampler_class, global_kernel):
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


@pytest.mark.local_only
@pytest.mark.parametrize(
    "sampler_class", [
        DiagonalJumpRWMH,
        DiagonalJumpMALA,
        DiagonalJumpHMC,
    ]
)
@pytest.mark.parametrize(
    "global_kernel", [
        'imh',
        'i-sir'
    ]
)
def test_nf_jump_diag_mh(sampler_class, global_kernel):
    torch.manual_seed(0)

    event_shape = (4,)
    n_chains = 50
    target = DiagonalGaussian(event_shape, mu=1.5, std=0.5)
    flow = Flow(RealNVP(event_shape, n_layers=1))

    sampler = sampler_class(
        flow=flow,
        neg_log_prob_target=target.neg_log_prob,
        global_kernel=global_kernel
    )

    z0 = torch.rand(size=(n_chains, *event_shape)) * 2 - 1
    _, latent_warmup_draws = sampler.warmup(
        z0=z0,
        n_steps=200,
        preconditioner_update_interval=50,
        return_latent_samples=True,
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
