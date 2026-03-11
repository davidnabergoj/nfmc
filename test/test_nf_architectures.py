import pytest
import torch

from nfmc.algorithms.preconditioning.samplers.neutra import NeuTraMALA, NeuTraRWMH
from test.util import DiagonalGaussian

from torchflows.flows import Flow
from torchflows.bijections.finite.autoregressive.architectures import RealNVP
from torchflows.bijections.finite.residual.architectures import ResFlow
from torchflows.bijections.continuous.rnode import RNODE


@pytest.mark.local_only
@pytest.mark.parametrize(
    "sampler_class", [
        NeuTraRWMH,
        NeuTraMALA
    ]
)
@pytest.mark.parametrize(
    "flow_class", [
        RealNVP,
        ResFlow,
        RNODE
    ]
)
def test_warmup_and_sample_no_runtime_error(sampler_class,
                                            flow_class):
    torch.manual_seed(0)

    event_shape = (2,)
    n_chains = 10
    target = DiagonalGaussian(event_shape, mu=1.5, std=0.5)

    flow = Flow(flow_class(event_shape))
    sampler = sampler_class(
        flow=flow,
        neg_log_prob_target=target.neg_log_prob
    )

    z0 = torch.rand(size=(n_chains, *event_shape)) * 2 - 1
    _, latent_warmup_draws = sampler.warmup(
        z0=z0,
        n_cycles=5,
        cycle_length=7,
        return_latent_samples=True,
        n_epochs=2
    )

    sampling_draws = sampler.sample(
        z0=latent_warmup_draws.last_sample,
        n_steps=3,
    )
