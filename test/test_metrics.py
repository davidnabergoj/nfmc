import torch
from nfmc.metrics import squared_bias_2nd_moment, squared_bias_mean


def test_squared_bias():
    n_steps = 200
    n_chains = 100
    n_dim = 2

    samples = torch.randn(size=(n_steps, n_chains, n_dim))
    ground_truth_mean = torch.zeros(size=(n_dim,))
    ground_truth_var = torch.ones(size=(n_dim,))
    ground_truth_second_moment = torch.ones(size=(n_dim,))

    ret = squared_bias_mean(samples, ground_truth_mean, ground_truth_var)
    assert ret.shape == (n_steps,)
    assert torch.all(~torch.isnan(ret))
    assert torch.all(~torch.isinf(ret))

    ret = squared_bias_2nd_moment(samples, ground_truth_second_moment, ground_truth_var)
    assert ret.shape == (n_steps,)
    assert torch.all(~torch.isnan(ret))
    assert torch.all(~torch.isinf(ret))
