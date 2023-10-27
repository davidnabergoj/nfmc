import torch


def mh_step(x: torch.Tensor, proposal_std: torch.Tensor = None):
    # metropolis hastings step with a diagonal normal proposal
    n_chains, n_dim = x.shape
    if proposal_std is None:
        proposal_std = torch.ones([1, n_dim], dtype=x.dtype)
    else:
        assert proposal_std.shape == (1, n_dim)
    noise = torch.randn_like(x) * proposal_std
    x_prime = x + noise

    # return proposed point and log of the metropolis acceptance ratio

    return x_prime


def mh(
        x: torch.Tensor,
        proposal_std: torch.Tensor = None,
        inv_mass_diag: torch.Tensor = None
):
    raise NotImplementedError
