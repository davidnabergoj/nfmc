import time

import pytest
import torch

from nfmc import sample
from nfmc.util import get_supported_samplers


@pytest.mark.skip(reason="May not terminate")
@pytest.mark.parametrize('strategy', get_supported_samplers())
def test_sampling(strategy: str):
    torch.manual_seed(0)
    t0 = time.time()
    sample(
        target=lambda x: torch.sum(x ** 2, dim=1),
        event_shape=(10,),
        n_iterations=1_000_000,
        sampling_time_limit_seconds=1.0,
        warmup=False
    )
    assert time.time() - t0 < 10.0


@pytest.mark.skip(reason="May not terminate")
@pytest.mark.parametrize('strategy', get_supported_samplers())
def test_warmup(strategy: str):
    torch.manual_seed(0)
    t0 = time.time()
    sample(
        target=lambda x: torch.sum(x ** 2, dim=1),
        event_shape=(10,),
        n_iterations=1_000_000,
        sampling_time_limit_seconds=1.0,
        warmup_time_limit_seconds=1.0,
        warmup=True
    )
    assert time.time() - t0 < 20.0
