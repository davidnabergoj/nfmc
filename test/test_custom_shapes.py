import pytest
import torch

from nfmc import sample


@pytest.mark.parametrize(
    'event_shape',
    [
        (8, 8),
        # (16, 16),
        # (32, 32),
        # (64, 64),
        # (128, 128),
    ]
)
@pytest.mark.parametrize(
    'strategy',
    [
        'imh', 'jump_mh', 'neutra_mh',
        'jump_hmc', 'neutra_hmc',
        'hmc',
        'mh'
    ]
)
def test_image(event_shape, strategy):
    torch.manual_seed(0)

    def target(x):
        return torch.sum(x ** 2, dim=(-2, -1))

    sample(
        target,
        event_shape=event_shape,
        strategy=strategy,
        n_iterations=2,
        n_warmup_iterations=2,
        n_chains=3,
        warmup=False,
        inner_param_kwargs=dict(n_iterations=2)
    )