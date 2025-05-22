import time
from typing import Tuple, Union
import torch
from tqdm import tqdm

from nfmc.algorithms.mh.data import MHOutput
from nfmc.algorithms.util.expectation import MCMCExpectation, MCMCExpectationDict


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
        self._n_divergences: int = 0   # Counts the number of chains that diverged across all steps

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
                 functional: callable = None, 
                 **kwargs):
        """
        MHSampler constructor.
        
        :param callable functional: estimate the expectation of this functional. The functional is applied to each drawn 
         sample and averaged on the fly. This does not affect drawn samples, but instead produces separate first and 
         second moments of the functional.
        """
        self.kernel = kernel

        self.expectations = MCMCExpectationDict(
            {
                'first_moment': MCMCExpectation(self.kernel.event_shape, f=lambda v: v),
                'second_moment': MCMCExpectation(self.kernel.event_shape, f=lambda v: v ** 2),
            },
            data_transform=functional
        )

    @property
    def name(self) -> str:
        return "Metropolis-Hastings sampler"

    @property
    def functional_first_moment(self):
        return self.expectations['first_moment'].as_tensor()
    
    @property
    def functional_second_moment(self):
        return self.expectations['second_moment'].as_tensor()
    
    @property
    def functional_variance(self):
        return self.functional_second_moment - self.functional_first_moment ** 2
    
    @property
    def acceptance_rate(self):
        if self.kernel._n_attempted_transitions == 0:
            return torch.nan
        return self.kernel._n_accepted_transitions / self.kernel._n_attempted_transitions

    @property
    def calls_per_second(self, elapsed_time_seconds):
        if elapsed_time_seconds > 0:
            return self.kernel.n_target_calls / elapsed_time_seconds
        return torch.nan

    @property
    def grads_per_second(self, elapsed_time_seconds):
        if elapsed_time_seconds > 0:
            return self.kernel.n_target_gradient_calls / elapsed_time_seconds
        return torch.nan

    def warmup(self,
               x0: torch.Tensor,
               show_progress: bool = True,
               time_limit_seconds: Union[float, int] = None) -> MHOutput:
        """
        Optimizes kernel parameters.
        """
        raise NotImplementedError

    def sample(self,
               x0: torch.Tensor,
               n_steps: int,
               show_progress: bool = True,
               time_limit_seconds: Union[float, int] = None,
               max_samples: int = None,
               data_transform: callable = None) -> MHOutput:
        """
        Draw samples with a fixed kernel.

        :param torch.Tensor x0: initial state with shape `(n_chains, *event_shape)`.
        :param int n_steps: number of MCMC steps to perform.
        :param bool show_progress: if True, display a progress bar.
        :param float time_limit_seconds: maximum sampling time. Sampling stops if this time is exceeded.
        """

        out = MHOutput(
            event_shape,
            max_samples=max_samples,
            data_transform=data_transform
        )
        out.statistics.data_transform = self.data_transform
        x = torch.clone(x0).detach()

        for _ in (pbar := tqdm(range(n_steps),
                               desc=f'Metropolis-Hastings sampling ({self.kernel.name} kernel)',
                               disable=not show_progress)):
            if time_limit_seconds is not None and out.statistics.elapsed_time_seconds > time_limit_seconds:
                break

            t0 = time.time()
            x_prime, mask = self.kernel.step(x)
            x[mask] = x_prime[mask]
            out.statistics.expectations.update(x)
            out.running_samples.add(x)

            pbar.set_postfix_str(f'{out.statistics} | {self.kernel}')

            out.statistics.update_elapsed_time(time.time() - t0)
            if out.statistics.elapsed_time_seconds > time_limit_seconds:
                break

        out.kernel = self.kernel
        return out
