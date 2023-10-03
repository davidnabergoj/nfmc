import torch

from nfmc.util import metropolis_acceptance_log_ratio
from potentials.synthetic.gaussian import StandardGaussian, DiagonalGaussian


def test_log_ratio():
    x0 = torch.tensor([[-100.0, -100.0]])
    x1 = torch.tensor([[0.0, 0.0]])

    target = StandardGaussian(event_shape=(2,))
    proposal = DiagonalGaussian(mu=torch.zeros(2), sigma=torch.ones(2) * 100)

    # Move from x0 to x1
    forward = metropolis_acceptance_log_ratio(
        log_prob_curr=-target(x0),
        log_prob_prime=-target(x1),
        log_proposal_curr=-proposal(x0),
        log_proposal_prime=-proposal(x1)
    )

    # Move from x1 to x0
    inverse = metropolis_acceptance_log_ratio(
        log_prob_curr=-proposal(x0),
        log_prob_prime=-proposal(x1),
        log_proposal_curr=-target(x0),
        log_proposal_prime=-target(x1)
    )

    assert forward > inverse