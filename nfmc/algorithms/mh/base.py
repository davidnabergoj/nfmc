from copy import deepcopy
import time
from typing import Tuple, Union
import torch
from tqdm import tqdm
from nfmc.algorithms.util.samples import Samples


class MHKernel:
    """
    Base MCMC kernel class for Metropolis-Hastings algorithms.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 neg_log_prob_target: callable):
        """
        MHKernel constructor.

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param callable target: function that computes the negative log probability density of the target distribution.
         It receives as input a tensor with shape `(*batch_shape, *event_shape)` and outputs a tensor with shape
         `batch_shape`.
        """
        self.event_shape = event_shape
        self.neg_log_prob_target = neg_log_prob_target

        self._n_steps: int = 0
        self._n_attempted_transitions: int = 0
        self._n_accepted_transitions: int = 0
        self._n_calls: int = 0  # Target density evaluation counter
        self._n_grads: int = 0  # Target density gradient evaluation counter
        # Counts the number of chains that diverged across all steps
        self._n_divergences: int = 0

    @property
    def name(self) -> str:
        raise NotImplementedError

    def reset_statistics(self):
        self._n_steps = 0
        self._n_attempted_transitions = 0
        self._n_accepted_transitions = 0
        self._n_calls = 0
        self._n_grads = 0
        self._n_divergences = 0

    @property
    def acceptance_rate(self):
        if self._n_attempted_transitions == 0:
            return torch.nan
        return self._n_accepted_transitions / self._n_attempted_transitions

    def increment_n_steps(self):
        self._n_steps += 1

    def increment_n_attempted_transitions(self, n_chains: int):
        self._n_attempted_transitions += n_chains
        self._n_attempted_transitions = int(self._n_attempted_transitions)

    def increment_n_accepted_transitions(self, n_accepted_chains: int):
        self._n_accepted_transitions += n_accepted_chains
        self._n_accepted_transitions = int(self._n_accepted_transitions)

    def increment_n_calls(self, n_calls: int):
        self._n_calls += n_calls
        self._n_calls = int(self._n_calls)

    def increment_n_grads(self, n_grads: int):
        self._n_grads += n_grads
        self._n_grads = int(self._n_grads)

    def increment_n_divergences(self, n_divergences: int):
        self._n_divergences += n_divergences
        self._n_divergences = int(self._n_divergences)

    @property
    def event_size(self):
        return int(torch.prod(torch.as_tensor(self.event_shape)))

    def step(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs one kernel transition.

        Note: the returned state tensor is the proposed state, not the new state after the accept/reject step.

        :param torch.Tensor x: current state tensor with shape `(n_chains, *event_shape)`.
        :return: proposed state tensor with shape `(n_chains, *event_shape)` and acceptance mask tensor with shape 
         `(n_chains)`.
        """
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


class MHSampler:
    """
    Sampler class for Metropolis-Hastings algorithms.
    """

    def __init__(self,
                 kernel: MHKernel,
                 **kwargs):
        """
        MHSampler constructor.

        :param MHKernel kernel: Metropolis-Hastings kernel that performs state transitions.
        """
        self.kernel = kernel

    @property
    def name(self) -> str:
        return "Metropolis-Hastings sampler"

    def calls_per_second(self, elapsed_time_seconds):
        if elapsed_time_seconds > 0:
            return self.kernel._n_calls / elapsed_time_seconds
        return torch.nan

    def grads_per_second(self, elapsed_time_seconds):
        if elapsed_time_seconds > 0:
            return self.kernel._n_grads / elapsed_time_seconds
        return torch.nan

    def sample(self,
               x0: torch.Tensor,
               n_steps: int,
               show_progress: bool = True,
               time_limit_seconds: Union[float, int] = None,
               max_samples: int = None,
               data_transform: callable = None) -> Samples:
        """
        Draw samples with a fixed kernel.

        :param torch.Tensor x0: initial state with shape `(n_chains, *event_shape)`.
        :param int n_steps: number of MCMC steps to perform.
        :param bool show_progress: if True, display a progress bar.
        :param float time_limit_seconds: maximum sampling time. Sampling stops if this time is exceeded.
        """

        samples = Samples(
            event_shape=self.kernel.event_shape,
            max_samples=max_samples,
            data_transform=data_transform
        )
        self.kernel.reset_statistics()
        x = deepcopy(x0.detach())

        t0 = time.time()
        for _ in (pbar := tqdm(range(n_steps),
                               desc=f'{self.kernel.name} sampling',
                               disable=not show_progress)):
            x_prime, mask = self.kernel.step(x)
            x[mask] = x_prime[mask]
            samples.add(x)

            elapsed_time = time.time() - t0
            pbar.set_postfix_str(
                f'acc-rate: {self.kernel.acceptance_rate} | '
                f'calls/s: {self.calls_per_second(elapsed_time)} | '
                f'grads/s: {self.grads_per_second(elapsed_time)} | '
            )
            if time_limit_seconds is not None and elapsed_time > time_limit_seconds:
                break

        return samples
