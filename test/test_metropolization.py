import torch

from nfmc.util import metropolis_acceptance_log_ratio
from test.util import standard_gaussian_potential, diagonal_gaussian_potential


def test_log_ratio():
    x0 = torch.tensor([[-100.0, -100.0]])
    x1 = torch.tensor([[0.0, 0.0]])

    target = standard_gaussian_potential
    proposal = diagonal_gaussian_potential

    # Move from x0 to x1
    forward = metropolis_acceptance_log_ratio(
        log_prob_target_curr=-target(x0),
        log_prob_target_prime=-target(x1),
        log_prob_proposal_curr=-proposal(x0),
        log_prob_proposal_prime=-proposal(x1)
    )

    # Move from x1 to x0
    inverse = metropolis_acceptance_log_ratio(
        log_prob_target_curr=-proposal(x0),
        log_prob_target_prime=-proposal(x1),
        log_prob_proposal_curr=-target(x0),
        log_prob_proposal_prime=-target(x1)
    )

    assert forward > inverse
