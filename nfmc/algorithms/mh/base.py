from copy import deepcopy
import time
from typing import Tuple, Union
import torch
from tqdm import tqdm

from nfmc.algorithms.kernel import MarkovKernel
from nfmc.algorithms.sampling.base.sampler import MCMCSampler
from nfmc.algorithms.util.samples import Samples


class MHKernel(MarkovKernel):
    """
    Base MCMC kernel class for Metropolis-Hastings algorithms.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 neg_log_prob_target: callable, 
                 **kwargs):
        """
        MHKernel constructor.

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param callable target: function that computes the negative log probability density of the target distribution.
         It receives as input a tensor with shape `(*batch_shape, *event_shape)` and outputs a tensor with shape
         `batch_shape`.
        """
        super().__init__(event_shape, neg_log_prob_target, **kwargs)

        self._n_attempted_transitions: int = 0
        self._n_accepted_transitions: int = 0

    def pbar_repr(self, elapsed_time_seconds: float):
        """
        Returns a string that represents this object in sampling/warmup progress bars.
        """
        data = [
            self.name,
            f'{self.calls_per_second(elapsed_time_seconds):.3f} c/s',
            f'{self.grads_per_second(elapsed_time_seconds):.3f} g/s',
            f'{self.acceptance_rate:.3f} acc',
        ]
        return ', '.join(data)

    def reset_statistics(self):
        super().reset_statistics()
        self._n_attempted_transitions = 0
        self._n_accepted_transitions = 0

    @property
    def acceptance_rate(self):
        if self._n_attempted_transitions == 0:
            return torch.nan
        return self._n_accepted_transitions / self._n_attempted_transitions

    def increment_n_attempted_transitions(self, n_chains: int):
        self._n_attempted_transitions += n_chains
        self._n_attempted_transitions = int(self._n_attempted_transitions)

    def increment_n_accepted_transitions(self, n_accepted_chains: int):
        self._n_accepted_transitions += n_accepted_chains
        self._n_accepted_transitions = int(self._n_accepted_transitions)


class MHSampler(MCMCSampler):
    """
    Sampler class for Metropolis-Hastings algorithms.
    """

    def __init__(self, kernel: MHKernel, **kwargs):
        """
        MHSampler constructor.

        :param MHKernel kernel: Metropolis-Hastings kernel that performs state transitions.
        """
        self.kernel = kernel

    @property
    def name(self) -> str:
        return "Metropolis-Hastings sampler"

    def warmup(self,
               x0: torch.Tensor,
               n_steps: int,
               show_progress: bool = True,
               time_limit_seconds: Union[float, int] = None,
               max_samples: int = None,
               data_transform: callable = None):
        self.kernel.start_warmup()
        out = self.sample(
            x0=x0,
            n_steps=n_steps,
            show_progress=show_progress,
            time_limit_seconds=time_limit_seconds,
            max_samples=max_samples,
            data_transform=data_transform,
            _tuning=True,
        )
        self.kernel.end_warmup()
        return out

    def sample(self,
               x0: torch.Tensor,
               n_steps: int,
               show_progress: bool = True,
               time_limit_seconds: Union[float, int] = None,
               max_samples: int = None,
               data_transform: callable = None,
               _tuning: bool = False) -> Samples:
        """
        Draw samples with a fixed kernel.

        :param torch.Tensor x0: initial state with shape `(*batch_shape, *event_shape)`.
        :param int n_steps: number of MCMC steps to perform.
        :param bool show_progress: if True, display a progress bar.
        :param float time_limit_seconds: maximum sampling time. Sampling stops if this time is exceeded.
        :param int max_samples: maximum number of samples to store.
        :param callable data_transform: function that transforms each generated sample. Receives as input a tensor with
         shape `(*batch_shape, *event_shape)` and outputs a tensor with shape `(*batch_shape, *event_shape)`.
        :param bool _tuning: if True, update the kernel at the end of each step.
        """
        samples = Samples(
            event_shape=self.kernel.event_shape,
            max_samples=max_samples,
            data_transform=data_transform,
        )

        self.kernel.reset_statistics()
        x = deepcopy(x0.detach())

        t0 = time.time()
        for _ in (
            pbar := tqdm(
                range(n_steps),
                desc=f"Sampling",
                disable=not show_progress,
            )
        ):
            x = self.kernel.step(x, update=_tuning)
            samples.add(x)

            elapsed_time = time.time() - t0
            pbar.set_postfix_str(self.kernel.pbar_repr(elapsed_time))
            if time_limit_seconds is not None and elapsed_time > time_limit_seconds:
                break

        return samples
