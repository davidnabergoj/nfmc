import math
from typing import Union, Tuple
import torch
from nfmc.algorithms.mh.base import MHKernel
from nfmc.algorithms.mh.local.dual_averaging import DualAveraging


class LocalMHKernel(MHKernel):
    """
    Metropolis-Hastings kernel with local transitions.
    Transitions use a step size.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 neg_log_prob_target: callable,
                 step_size: Union[float, torch.Tensor] = 0.01,
                 dual_averaging_kwargs: dict = None,
                 target_acceptance_rate: float = 0.651,
                 **kwargs):
        """
        LocalMHKernel constructor.

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param callable neg_log_prob_target: negative log probability density function. Takes as input a tensor with
         shape `(*batch_shape, *event_shape)` and returns a tensor with shape `batch_shape`.
        :param float step_size: positive step size float or tensor with shape `()`.
        :param torch.Tensor inv_mass_diag: inverse of the diagonal mass matrix. If None, the mass matrix is set to
         identity.
        :param dict dual_averaging_kwargs: keyword arguments passed to DualAveraging.
        :param float target_acceptance_rate: scalar between 0 and 1 (exclusive). Used in step size tuning via dual
         averaging.
        :param int mass_matrix_update_interval: number of kernel transitions before each mass matrix update.
        """
        super().__init__(event_shape, neg_log_prob_target, **kwargs)

        # Store initial step size
        step_size = torch.as_tensor(step_size)
        self._initial_step_size = step_size
        self._initial_dual_averaging_kwargs = dual_averaging_kwargs

        self.step_size = self._initial_step_size
        self._dual_averaging: DualAveraging = None
        self._target_acceptance_rate: float = target_acceptance_rate

    def _create_dual_averaging_object(self, n_chains: int):
        self._dual_averaging: DualAveraging = DualAveraging(
            self.step_size,
            n_chains=n_chains,
            **(self._initial_dual_averaging_kwargs or {})
        )

    def start_warmup(self, n_chains: int):
        super().start_warmup(n_chains=n_chains)
        self._create_dual_averaging_object(self._n_warmup_chains)

    def reset_parameters(self):
        self.step_size = self._initial_step_size
        self._create_dual_averaging_object(n_chains=self._n_warmup_chains)

    def finalize_parameters(self):
        """
        Use the weighted average of observed step sizes after warmup.
        """
        if len(self._dual_averaging.error_history) > 0:
            self.step_size = self._dual_averaging.weighted_value().mean()

    def pbar_repr(self, elapsed_time_seconds: float):
        if self._dual_averaging is not None:
            eps_mean = torch.mean(torch.as_tensor(self._dual_averaging.error_sum))
            eps_max = torch.max(torch.as_tensor(self._dual_averaging.error_sum))
            eps_min = torch.min(torch.as_tensor(self._dual_averaging.error_sum))
            da_str = f'DA[{eps_mean:.2f} ^{eps_max:.2f} v{eps_min:.2f}]'
        else:
            da_str = 'DA[None]'
        data = [
            self.name,
            f'log step: {torch.log(self.step_size):.3f}',
            da_str,
            f'{self.calls_per_second(elapsed_time_seconds):.3f} c/s',
            f'{self.grads_per_second(elapsed_time_seconds):.3f} g/s',
            f'{self.acceptance_rate:.3f} acc',
            f'{self._n_divergences} divs',
        ]
        return ', '.join(data)

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

        :param torch.Tensor m: acceptance mask tensor after kernel transition with dtype `torch.bool`.
        """
        accepted_mask = m.float()
        self._dual_averaging.step(
            self._target_acceptance_rate - accepted_mask
        )
        self.step_size = torch.mean(self._dual_averaging.value)
