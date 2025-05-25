import torch
from nfmc.algorithms.iterated_sir import IteratedSIRKernel
from test.util import StandardGaussian


def test_step_gaussian_proposal():
    torch.manual_seed(0)

    event_shape = (2, 3, 4)
    n_chains = 5

    x0 = torch.zeros(size=(n_chains, *event_shape))
    kernel = IteratedSIRKernel(
        event_shape=event_shape,
        neg_log_prob_target=StandardGaussian(event_shape).neg_log_prob
    )
    x_new = kernel.step(x0)

    assert x_new.shape == x0.shape
    assert x_new.dtype == x0.dtype
    assert torch.isfinite(x_new).all()
