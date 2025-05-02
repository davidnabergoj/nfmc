import torch

from nfmc.algorithms.sampling.mcmc.mh import MH, MHParameters
from nfmc.algorithms.sampling.nfmc.ex2mcmc import Ex2MCMC, Ex2MCMCParameters


def configure_sampler(event_shape, n_jumps, n_mh_steps_per_jump, n_mh_warmup_steps):
    def target(x):
        return torch.sum(x ** 2, dim=-1)

    sampler = Ex2MCMC(
        event_shape,
        target,
        inner_sampler=MH(
            event_shape,
            target,
            params=MHParameters(n_iterations=n_mh_steps_per_jump, n_warmup_iterations=n_mh_warmup_steps)
        ),
        params=Ex2MCMCParameters(n_iterations=n_jumps, n_warmup_iterations=n_jumps)
    )

    return sampler

def test_sampling():
    torch.manual_seed(0)
    event_shape = (2,)
    n_chains = 10
    n_jumps = 3
    n_mh_steps_per_jump = 4

    sampler = configure_sampler(event_shape, n_jumps, n_mh_steps_per_jump, 0)
    out = sampler.sample(torch.randn(n_chains, *event_shape))

    assert out.samples.shape == (n_jumps + n_mh_steps_per_jump * n_jumps, n_chains, *event_shape)
    assert torch.isfinite(out.samples).all()

def test_sampling_no_local():
    torch.manual_seed(0)
    event_shape = (2,)
    n_chains = 10
    n_jumps = 3
    n_mh_steps_per_jump = 0

    sampler = configure_sampler(event_shape, n_jumps, n_mh_steps_per_jump, 0)
    out = sampler.sample(torch.randn(n_chains, *event_shape))

    assert out.samples.shape == (n_jumps, n_chains, *event_shape)
    assert torch.isfinite(out.samples).all()

def test_warmup():
    torch.manual_seed(0)
    event_shape = (2,)
    n_chains = 10
    n_jumps = 3
    n_mh_steps_per_jump = 4
    n_mh_warmup_steps = 5

    sampler = configure_sampler(event_shape, n_jumps, n_mh_steps_per_jump, n_mh_warmup_steps)
    out = sampler.warmup(torch.randn(n_chains, *event_shape))

    assert out.samples.shape == (n_mh_warmup_steps, n_chains, *event_shape)
    assert torch.isfinite(out.samples).all()

def test_warmup_no_local():
    torch.manual_seed(0)
    event_shape = (2,)
    n_chains = 10
    n_jumps = 3
    n_mh_steps_per_jump = 0
    n_mh_warmup_steps = 0

    sampler = configure_sampler(event_shape, n_jumps, n_mh_steps_per_jump, n_mh_warmup_steps)
    out = sampler.warmup(torch.randn(n_chains, *event_shape))

    assert out.samples.shape == (0, 0, *event_shape)
    assert torch.isfinite(out.samples).all()

def test_warmup_local_and_sampling_no_local():
    torch.manual_seed(0)
    event_shape = (2,)
    n_chains = 10
    n_jumps = 3
    n_mh_steps_per_jump = 0
    n_mh_warmup_steps = 20

    sampler = configure_sampler(event_shape, n_jumps, n_mh_steps_per_jump, n_mh_warmup_steps=n_mh_warmup_steps)
    w_out = sampler.warmup(torch.randn(n_chains, *event_shape))
    s_out = sampler.sample(w_out.running_samples.last_sample)

    assert s_out.samples.shape == (n_jumps, n_chains, *event_shape)
    assert torch.isfinite(s_out.samples).all()