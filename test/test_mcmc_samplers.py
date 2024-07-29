from nfmc.mcmc import mh, mala, hmc, nuts
import torch


def test_mh():
    torch.manual_seed(0)

    n_steps = 5
    n_chains = 100
    n_dim = 10

    x0 = torch.randn(size=(n_chains, n_dim))
    out = mh(x0, lambda x: torch.sum(x ** 2, dim=1), n_iterations=n_steps)

    assert out.shape == (n_steps, n_chains, n_dim)
    assert torch.all(torch.isfinite(out))


def test_mala():
    torch.manual_seed(0)

    n_steps = 5
    n_chains = 100
    n_dim = 10

    x0 = torch.randn(size=(n_chains, n_dim))
    out = mala(x0, lambda x: torch.sum(x ** 2, dim=1), n_iterations=n_steps)

    assert out.shape == (n_steps, n_chains, n_dim)
    assert torch.all(torch.isfinite(out))


def test_hmc():
    torch.manual_seed(0)

    n_steps = 5
    n_chains = 100
    n_dim = 10

    x0 = torch.randn(size=(n_chains, n_dim))
    out = hmc(x0, lambda x: torch.sum(x ** 2, dim=1), n_iterations=n_steps)

    assert out.shape == (n_steps, n_chains, n_dim)
    assert torch.all(torch.isfinite(out))


def test_nuts():
    torch.manual_seed(0)

    n_steps = 5
    n_dim = 10

    out = nuts(n_dim, lambda x: torch.sum(x ** 2, dim=1), n_steps)

    assert out.shape == (n_steps, 1, n_dim)
    assert torch.all(torch.isfinite(out))
