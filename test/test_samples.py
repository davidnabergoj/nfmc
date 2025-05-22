import torch
from nfmc.algorithms.util.samples import Samples


def test_add_single_step():
    n_chains = 10
    event_shape = (2, 3, 5)
    s = Samples(event_shape=event_shape)

    x = torch.zeros(size=(n_chains, *event_shape))
    s.add(x)

    assert s.n_samples == 1
    assert s.as_tensor().shape == (1, n_chains, *event_shape)


def test_add_three_single_steps():
    n_chains = 10
    event_shape = (2, 3, 5)
    s = Samples(event_shape=event_shape)

    for _ in range(3):
        x = torch.zeros(size=(n_chains, *event_shape))
        s.add(x)

    assert s.n_samples == 3
    assert s.as_tensor().shape == (3, n_chains, *event_shape)


def test_add_multiple_steps():
    n_steps = 7
    n_chains = 10
    event_shape = (2, 3, 5)
    s = Samples(event_shape=event_shape)

    x = torch.zeros(size=(n_steps, n_chains, *event_shape))
    s.add(x)

    assert s.n_samples == n_steps
    assert s.as_tensor().shape == (n_steps, n_chains, *event_shape)


def test_add_three_multiple_steps():
    n_steps = 7
    n_chains = 10
    event_shape = (2, 3, 5)
    s = Samples(event_shape=event_shape)

    for _ in range(3):
        x = torch.zeros(size=(n_steps, n_chains, *event_shape))
        s.add(x)

    assert s.n_samples == n_steps * 3
    assert s.as_tensor().shape == (n_steps * 3, n_chains, *event_shape)


def test_as_tensor_empty():
    event_shape = (2, 3, 5)
    s = Samples(event_shape=event_shape)
    assert s.as_tensor().shape == (0, 0, *event_shape)


def test_last_sample():
    n_steps = 7
    n_chains = 10
    event_shape = (2, 3, 5)
    s = Samples(event_shape=event_shape)

    for _ in range(3):
        x = torch.zeros(size=(n_steps, n_chains, *event_shape))
        s.add(x)

    assert s.last_sample.shape == (n_chains, *event_shape)


def test_last_sample_empty():
    event_shape = (2, 3, 5)
    s = Samples(event_shape=event_shape)

    assert s.last_sample is None
    assert s.n_samples == 0


def test_n_samples_empty():
    event_shape = (2, 3, 5)
    s = Samples(event_shape=event_shape)

    assert s.n_samples == 0


def test_reservoir_limit():
    reservoir_limit = 5

    n_steps = 7
    n_chains = 10
    event_shape = (2, 3, 5)
    s = Samples(event_shape=event_shape, max_samples=reservoir_limit)

    x = torch.zeros(size=(n_steps, n_chains, *event_shape))
    s.add(x)

    assert s.n_samples == reservoir_limit
    assert s.as_tensor().shape == (reservoir_limit, n_chains, *event_shape)


def test_reservoir_limit_one():
    reservoir_limit = 1

    n_steps = 7
    n_chains = 10
    event_shape = (2, 3, 5)
    s = Samples(event_shape=event_shape, max_samples=reservoir_limit)

    x = torch.zeros(size=(n_steps, n_chains, *event_shape))
    s.add(x)

    assert s.n_samples == reservoir_limit
    assert s.as_tensor().shape == (reservoir_limit, n_chains, *event_shape)


def test_functional():
    n_chains = 10
    event_shape = (2, 3, 5)
    s = Samples(event_shape=event_shape, data_transform=lambda v: v ** 2)

    x = torch.zeros(size=(n_chains, *event_shape)) + 2
    s.add(x)

    assert s.n_samples == 1
    assert s.as_tensor().shape == (1, n_chains, *event_shape)
    assert torch.all(s.as_tensor() == 4.0)


def test_moments():
    n_chains = 10
    event_shape = (2, 3, 5)
    s = Samples(event_shape=event_shape)

    x = torch.zeros(size=(n_chains, *event_shape)) + 2
    s.add(x)

    assert torch.all(s.first_moment.as_tensor() == 2.0)
    assert torch.all(s.second_moment.as_tensor() == 4.0)
