import torch

from nfmc import sample
from test.util import standard_gaussian_potential


def test_basic():
    torch.manual_seed(0)

    out_basic = sample(
        event_shape=(100,),
        target=standard_gaussian_potential,
        flow='realnvp',
        strategy='imh',
        n_iterations=3,
        n_warmup_iterations=3
    )
    n_basic_layers = len(out_basic.kernel.flow.bijection.layers)

    out_advanced = sample(
        event_shape=(100,),
        target=standard_gaussian_potential,
        flow='realnvp%{"n_layers": 10}',
        strategy='imh',
        n_iterations=3,
        n_warmup_iterations=3
    )
    n_advanced_layers = len(out_advanced.kernel.flow.bijection.layers)

    assert n_advanced_layers > n_basic_layers


def test_advanced():
    torch.manual_seed(0)

    out_basic = sample(
        event_shape=(100,),
        target=standard_gaussian_potential,
        flow='realnvp',
        strategy='imh',
        n_iterations=3,
        n_warmup_iterations=3
    )
    n_basic_layers = len(out_basic.kernel.flow.bijection.layers)

    out_advanced = sample(
        event_shape=(100,),
        target=standard_gaussian_potential,
        flow='realnvp%{"n_layers": 10, "conditioner_kwargs": {"n_layers": 5, "n_hidden": 100}}',
        strategy='imh',
        n_iterations=3,
        n_warmup_iterations=3
    )
    n_advanced_layers = len(out_advanced.kernel.flow.bijection.layers)

    assert n_advanced_layers > n_basic_layers
