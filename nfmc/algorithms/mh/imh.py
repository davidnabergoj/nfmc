from typing import Tuple, Union
import torch
from nfmc.algorithms.mh.base import MHKernel
from nfmc.util import compute_divergence_mask, metropolis_acceptance_log_ratio, sum_except_batch


class IMHKernel(MHKernel):
    """
    Independent Metropolis-Hastings kernel.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 neg_log_prob_target: callable,
                 proposal_log_prob: callable = None,
                 proposal_sample_with_log_prob: callable = None, 
                 **kwargs):
        """
        IMH kernel constructor.

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event.
        :param callable neg_log_prob_target: function that computes the negative of the log target probability density. 
         Receives as input a tensor with shape `(*batch_shape, *event_shape)` and outputs a tensor with shape 
         `batch_shape`.
        :param callable proposal_log_prob: function that computes the unnormalized log proposal probability density. 
         Receives as input a tensor with shape `(*batch_shape, *event_shape)` and outputs a tensor with shape 
         `batch_shape`. If None, use a standard Gaussian proposal.
        :param callable proposal_sample_with_log_prob: function that draws samples from the proposal distribution. 
         Receives as input a sample shape tuple `batch_shape` and returns a tensor with shape 
         `(*batch_shape, *event_shape)`, and the corresponding log probability density tensor with shape `batch_shape`.
         If None, use a standard Gaussian proposal.
        """
        super().__init__(event_shape, neg_log_prob_target, **kwargs)

        if proposal_log_prob is None and proposal_sample_with_log_prob is not None:
            raise ValueError(
                "Both or neither of proposal_log_prob and proposal_sample_with_log_prob must be provided")
        if proposal_log_prob is not None and proposal_sample_with_log_prob is None:
            raise ValueError(
                "Both or neither of proposal_log_prob and proposal_sample_with_log_prob must be provided")
        if proposal_log_prob is None and proposal_sample_with_log_prob is None:
            dist = torch.distributions.Normal(
                loc=torch.zeros(size=event_shape),
                scale=torch.ones(size=event_shape)
            )

            def _prop_lp(_in):
                return sum_except_batch(dist.log_prob(_in), event_shape)

            def _prop_swlp(batch_shape):
                _x = dist.sample(sample_shape=batch_shape)
                _lp = _prop_lp(_x)
                return _x, _lp

            proposal_log_prob = _prop_lp
            proposal_sample_with_log_prob = _prop_swlp

        self.proposal_log_prob = proposal_log_prob
        self.proposal_sample_with_log_prob = proposal_sample_with_log_prob

    def reset_parameters(self):
        pass  # Nothing to reset

    @property
    def name(self):
        return 'IMH'

    def step(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Perform one IMH transition.

        :param torch.Tensor x: incoming state tensor with shape `(*batch_shape, *event_shape)`.
        :return: new state tensor with shape `(*batch_shape, *event_shape)`.
        """
        # Propose new state
        batch_shape = x.shape[:-len(self.event_shape)]
        u_x = -self.proposal_log_prob(x)
        x_prime, log_prob_x_prime = self.proposal_sample_with_log_prob(
            batch_shape
        )
        x_prime = x_prime.to(x)
        log_prob_x_prime = log_prob_x_prime.to(x)
        u_x_prime = -log_prob_x_prime

        # Compute divergence mask
        divergence_mask = compute_divergence_mask(x_prime, self.event_shape)
        n_valid_proposals = int((~divergence_mask).long().sum())
        self.increment_n_divergences(int(divergence_mask.long().sum()))
        self.increment_n_divergences_per_chain(divergence_mask)

        # Compute acceptance mask
        acceptance_mask = torch.zeros_like(divergence_mask)
        log_prob_accept = metropolis_acceptance_log_ratio(
            -self.neg_log_prob_target(x[~divergence_mask]),
            -self.neg_log_prob_target(x_prime[~divergence_mask]),
            -u_x[~divergence_mask],
            -u_x_prime[~divergence_mask],
        )
        self.increment_n_calls(n_valid_proposals * 2)

        log_u = torch.rand_like(log_prob_accept).log()
        acceptance_mask[~divergence_mask] = log_u < log_prob_accept
        x_new = x.clone()
        x_new[acceptance_mask] = x_prime[acceptance_mask]

        self.increment_n_steps()
        self.increment_n_attempted_transitions(n_chains=x.shape[0])
        self.increment_n_accepted_transitions(
            int(acceptance_mask.long().sum()))

        return x_new
