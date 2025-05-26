import pytest
import torch

from torchflows.flows import Flow
from torchflows.architectures import RealNVP

from nfmc.algorithms.mh.local.hmc import HMCKernel
from nfmc.algorithms.mh.local.rwmh import RWMHKernel
from nfmc.algorithms.mh.local.mala import MALAKernel
from nfmc.algorithms.mh.local.base import LocalMHSampler
from nfmc.algorithms.preconditioning.base import PreconditionedMCMCSampler
from nfmc.algorithms.preconditioning.implementations import (
    NeuTraHMC,
    NeuTraMALA,
    NeuTraRWMH,
)

from test.util import DiagonalGaussian


@pytest.mark.parametrize("kernel_class", [RWMHKernel, MALAKernel, HMCKernel])
@pytest.mark.local_only
def test_mcmc(kernel_class):
    torch.manual_seed(0)

    event_shape = (4,)
    target = DiagonalGaussian(event_shape)

    kernel = kernel_class(event_shape, neg_log_prob_target=target.neg_log_prob)
    sampler = LocalMHSampler(kernel)

    x0 = torch.rand(size=(1, *event_shape)) * 2 - 1
    warmup_draws = sampler.warmup(x0=x0, n_steps=1000)
    sampling_draws = sampler.sample(x0=warmup_draws.last_sample, n_steps=2000)

    assert torch.allclose(
        sampling_draws.first_moment.as_tensor(), target.first_moment, rtol=0.2
    )
    assert torch.allclose(
        sampling_draws.second_moment.as_tensor(), target.second_moment, rtol=0.2
    )


@pytest.mark.local_only
@pytest.mark.parametrize(
    "sampler_class", [NeuTraRWMH, NeuTraMALA, NeuTraHMC]
)
def test_preconditioned_mcmc(sampler_class):
    torch.manual_seed(0)

    event_shape = (4,)
    n_chains = 4
    target = DiagonalGaussian(event_shape)
    flow = Flow(RealNVP(event_shape))

    sampler = sampler_class(
        flow=flow,
        neg_log_prob_target=target.neg_log_prob
    )

    z0 = torch.rand(size=(n_chains, *event_shape)) * 2 - 1
    warmup_draws, latent_warmup_draws = sampler.warmup(
        z0=z0,
        n_steps=1200,
        preconditioner_update_interval=500,
        return_latent_samples=True,
    )
    sampling_draws = sampler.sample(
        z0=latent_warmup_draws.last_sample, n_steps=2000)

    flow_samples = flow.sample((10000,)).detach()
    flow_first_moment = flow_samples.mean(0)
    flow_second_moment = flow_samples.square().mean(0)

    assert torch.allclose(target.first_moment, flow_first_moment, rtol=0.2)
    assert torch.allclose(target.second_moment, flow_second_moment, rtol=0.2)

    assert torch.allclose(
        sampling_draws.first_moment.as_tensor(), target.first_moment, rtol=0.2
    )
    assert torch.allclose(
        sampling_draws.second_moment.as_tensor(), target.second_moment, rtol=0.2
    )


@pytest.mark.local_only
def test_jump_mcmc():
    torch.manual_seed(0)
