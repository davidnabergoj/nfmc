import torch
from nfmc.algorithms.kernel import MarkovKernel
from nfmc.util import sum_except_batch


class IteratedSIRKernel(MarkovKernel):
    """
    Iterated sampling importance resampling kernel class.

    At each step, several candidate states are drawn from a proposal distribution. A weight is calculated for each
    candidate state, including the current state. The new state is chosen by sampling from a categorical distribution,
    defined by candidate weights.
    """

    def __init__(self,
                 event_shape,
                 neg_log_prob_target,
                 proposal_log_prob: callable = None,
                 proposal_sample_with_log_prob: callable = None,
                 pool_size: int = 20, 
                 **kwargs):
        """
        IteratedSIRKernel constructor.

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
        :param int pool_size: number of candidate states in each step, including the current state.
        """
        super().__init__(event_shape, neg_log_prob_target, **kwargs)

        if proposal_log_prob is None and proposal_sample_with_log_prob is not None:
            raise ValueError(
                "Both proposal_log_prob and proposal_sample_with_log_prob must be provided")
        if proposal_log_prob is not None and proposal_sample_with_log_prob is None:
            raise ValueError(
                "Both proposal_log_prob and proposal_sample_with_log_prob must be provided")
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

        self.pool_size = pool_size

    @property
    def name(self):
        return "i-SIR"

    def step(self,
             x: torch.Tensor,
             update: bool = False,
             **kwargs):
        x_flat = x.view(-1, *self.event_shape)
        batch_size = len(x_flat)

        # Prepare tensors for candidates and log prob under the NF
        # candidates.shape = (batch_size, *event_shape, pool_size)
        candidates = torch.zeros(
            size=(batch_size, self.pool_size, *self.event_shape)
        ).to(x)
        log_prob_flow = torch.zeros(size=(batch_size, self.pool_size)).to(x)

        # sample candidates, compute log density under the target and the NF
        candidates[:, 0], log_prob_flow[:, 0] = (
            x,
            self.proposal_log_prob(x)
        )
        candidates[:, 1:], log_prob_flow[:, 1:] = self.proposal_sample_with_log_prob(
            (batch_size, self.pool_size - 1)
        )
        candidates = candidates.detach()
        log_prob_flow = log_prob_flow.detach()
        log_prob_target = -self.neg_log_prob_target(candidates)

        self.increment_n_steps()
        self.increment_n_calls(self.pool_size * batch_size)

        # compute self-normalized weights and sample candidates
        log_sn_weights = log_prob_target - log_prob_flow
        pool_indices = torch.tensor([
            int(
                torch.distributions.Categorical(
                    logits=log_sn_weights[i]
                ).sample(())
            )
            for i in range(len(log_sn_weights))
        ])

        # return chosen points
        return candidates[range(batch_size), pool_indices].view_as(x)
