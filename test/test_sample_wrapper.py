import pytest
from nfmc.sample import sample
import torch


@pytest.mark.parametrize('kernel', ['rwmh', 'mala', 'hmc'])
@pytest.mark.parametrize('n_chains', [1, 4])
def test_mcmc(kernel, n_chains):
    event_shape = (2,)
    out = sample(
        neg_log_prob_target=lambda v: torch.sum(v ** 2, dim=-1),
        event_shape=event_shape,
        kernel=kernel,
        warmup=True,
        n_sampling_steps=2,
        warmup_kwargs={
            'n_steps': 2,
        },
        n_chains=n_chains,
        show_progress=True
    )

    assert out.n_samples == 2


@pytest.mark.parametrize('kernel', [
    'imh',
    'i-sir',
    'jump_hmc',
    'jump_rwmh',
    'jump_mala',
    'neutra_hmc',
    'neutra_rwmh',
    'neutra_mala',
    'ex2_hmc',
    'ex2_rwmh',
    'ex2_mala',
])
@pytest.mark.parametrize('n_chains', [1, 4])
def test_preconditioned_mcmc(kernel, n_chains):
    event_shape = (5,)
    out = sample(
        neg_log_prob_target=lambda v: torch.sum(v ** 2, dim=-1),
        event_shape=event_shape,
        kernel=kernel,
        flow='realnvp',
        warmup=True,
        n_sampling_steps=2,
        warmup_kwargs={
            'n_cycles': 1,
            'cycle_length': 2
        },
        n_chains=n_chains,
        show_progress=True
    )

    assert out.n_samples == 2
