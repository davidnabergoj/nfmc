from copy import deepcopy
from tqdm import tqdm
import torch

from nfmc.util import metropolis_acceptance_log_ratio
from normalizing_flows import Flow


def sample_bounded_geom(p, max_val):
    v = torch.arange(0, max_val + 1)
    pdf = p * (1 - p) ** (max_val - v) / (1 - (1 - p) ** (max_val + 1))
    cdf = torch.cumsum(pdf, dim=0)
    assert torch.isclose(cdf[-1], torch.tensor(1.0))
    u = float(torch.rand(size=()))
    return int(torch.searchsorted(cdf, u, right=True))


def imh(x0: torch.Tensor,
        flow: Flow,
        potential: callable,
        n_iterations: int = 1000,
        full_output: bool = False,
        adaptation_dropoff: float = 0.9999,
        train_dist: str = 'uniform'):
    assert train_dist in ['bounded_geom_approx', 'bounded_geom', 'uniform']
    # Exponentially diminishing adaptation probability sequence
    xs_all = []
    xs_accepted = []

    n_accepted = 0
    n_total = 0

    n_chains, n_dim = x0.shape
    x = deepcopy(x0)
    for i in (pbar := tqdm(range(n_iterations))):
        x_proposed = flow.sample(n_chains, no_grad=True)
        log_alpha = metropolis_acceptance_log_ratio(
            log_prob_curr=-potential(x),
            log_prob_prime=-potential(x_proposed),
            log_proposal_curr=flow.log_prob(x),
            log_proposal_prime=flow.log_prob(x_proposed)
        )
        log_u = torch.rand(n_chains).log().to(log_alpha)
        accepted_mask = torch.less(log_u, log_alpha)
        x[accepted_mask] = x_proposed[accepted_mask]
        x = x.detach()

        xs_all.append(deepcopy(x))
        xs_accepted.append(deepcopy(x[accepted_mask]))

        n_accepted += int(torch.sum(accepted_mask.long()))
        n_total += n_chains
        instantaneous_acceptance_rate = float(torch.mean(accepted_mask.float()))

        u_prime = torch.rand(size=())
        alpha_prime = adaptation_dropoff ** i
        if u_prime < alpha_prime:
            # only use recent states to adapt
            # this is an approximation of a bounded "geometric distribution" that picks the training data
            # we can program the exact bounded geometric as well. Then it's parameter p can be adapted with dual
            # averaging.
            if train_dist == 'uniform':
                k = int(torch.randint(low=0, high=len(xs_all), size=()))
            elif train_dist == 'bounded_geom_approx':
                k = int(torch.randint(low=max(0, len(xs_all) - 100), high=len(xs_all), size=()))
            elif train_dist == 'bounded_geom':
                k = sample_bounded_geom(p=0.025, max_val=len(xs_all) - 1)
            else:
                raise ValueError
            x_train = xs_all[k]
            flow.fit(x_train, n_epochs=1)

        pbar.set_postfix_str(f'Running acceptance rate: {n_accepted / n_total:.3f}')

    xs_all = torch.stack(xs_all, dim=0)
    xs_accepted = torch.cat(xs_accepted, dim=0)

    if full_output:
        return xs_all, xs_accepted
    else:
        return x
