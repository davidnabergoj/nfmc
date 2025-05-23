from copy import deepcopy
import time
from typing import Union, Tuple
import torch
from tqdm import tqdm
from nfmc.algorithms.mh.base import MHKernel
from nfmc.algorithms.mh.local.dual_averaging import DualAveraging
from nfmc.algorithms.util.samples import Samples


class LocalMHKernel(MHKernel):
    """
    Metropolis-Hastings kernel with local transitions.
    Transitions use a step size.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 neg_log_prob_target: callable,
                 step_size: float = 0.01,
                 dual_averaging_kwargs: dict = None,
                 target_acceptance_rate: float = 0.651):
        """
        LocalMHKernel constructor.

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param callable neg_log_prob_target: negative log probability density function. Takes as input a tensor with 
         shape `(*batch_shape, *event_shape)` and returns a tensor with shape `batch_shape`.
        :param float step_size: positive step size.
        :param torch.Tensor inv_mass_diag: inverse of the diagonal mass matrix. If None, the mass matrix is set to
         identity.
        :param dict dual_averaging_kwargs: keyword arguments passed to DualAveraging.
        :param float target_acceptance_rate: scalar between 0 and 1 (exclusive). Used in step size tuning via dual 
         averaging.
        :param int mass_matrix_update_interval: number of kernel transitions before each mass matrix update.
        """
        super().__init__(event_shape, neg_log_prob_target)

        self.step_size = step_size
        self._dual_averaging: DualAveraging = DualAveraging(
            self.step_size,
            **(dual_averaging_kwargs or {})
        )
        self._target_acceptance_rate: float = target_acceptance_rate

    def step(self,
             x: torch.Tensor,
             update: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs one kernel transition.

        Note: the returned state tensor is the proposed state, not the new state after the accept/reject step.

        :param torch.Tensor x: current state tensor with shape `(n_chains, *event_shape)`.
        :param bool update: if True, also update the parameters of this kernel.
        :return: proposed state tensor with shape `(n_chains, *event_shape)` and acceptance mask tensor with shape 
         `(n_chains)`.
        """
        raise NotImplementedError

    def _update(self, m: torch.Tensor):
        """
        Update kernel parameters.

        :param torch.Tensor x: state tensor after kernel transition.
        :param torch.Tensor m: acceptance mask tensor after kernel transition with dtype `torch.bool`.
        :param bool tune_step_size: if True, update the step size whenever `update=True` in the `.step` method.
        :param bool tune_inv_mass_diag: if True, update the mass matrix whenever `update=True` in the `.step` method.
        """
        acc_rate = m.float().mean()
        error = self._target_acceptance_rate - acc_rate
        self._dual_averaging.step(error)
        self.step_size = self._dual_averaging.value


class LocalMHSampler:
    """
    Sampler class for local Metropolis-Hastings algorithms.
    """

    def __init__(self,
                 kernel: LocalMHKernel,
                 **kwargs):
        """
        MHSampler constructor.

        :param LocalMHKernel kernel: Metropolis-Hastings kernel that performs state transitions.
        """
        self.kernel = kernel

    @property
    def name(self) -> str:
        return "Local Metropolis-Hastings sampler"

    def calls_per_second(self, elapsed_time_seconds):
        if elapsed_time_seconds > 0:
            return self.kernel._n_calls / elapsed_time_seconds
        return torch.nan

    def grads_per_second(self, elapsed_time_seconds):
        if elapsed_time_seconds > 0:
            return self.kernel._n_grads / elapsed_time_seconds
        return torch.nan

    def warmup(self,
               x0: torch.Tensor,
               n_steps: int,
               show_progress: bool = True,
               time_limit_seconds: Union[float, int] = None,
               max_samples: int = None,
               data_transform: callable = None):
        return self.sample(
            x0=x0,
            n_steps=n_steps,
            show_progress=show_progress,
            time_limit_seconds=time_limit_seconds,
            max_samples=max_samples,
            data_transform=data_transform,
            _tuning=True,
        )

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

        :param torch.Tensor x0: initial state with shape `(n_chains, *event_shape)`.
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
            data_transform=data_transform
        )
        self.kernel.reset_statistics()
        x = deepcopy(x0.detach())

        t0 = time.time()
        for _ in (pbar := tqdm(range(n_steps),
                               desc=f'{self.kernel.name} sampling',
                               disable=not show_progress)):
            x = self.kernel.step(x, update=_tuning)
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