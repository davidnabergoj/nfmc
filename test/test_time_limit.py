import time
import torch

from nfmc import sample


def test_sampling():
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


def test_warmup():
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
