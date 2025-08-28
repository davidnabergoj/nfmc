from typing import Tuple, Union, List
import math
import warnings
import numpy as np
import torch

from nfmc.algorithms.preconditioning.preconditioners import Preconditioner
from nfmc.algorithms.preconditioning.preconditioners import IdentityPreconditioner


class MarkovKernel:
    """
    Base MCMC kernel class.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 neg_log_prob_target: callable,
                 preconditioner: Preconditioner = None):
        """
        MarkovKernel constructor.

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param callable target: function that computes the negative log probability density of the target distribution.
         It receives as input a tensor with shape `(*batch_shape, *event_shape)` and outputs a tensor with shape
         `batch_shape`.
        :param Preconditioner preconditioner: preconditioner for this kernel. If None, uses the identity preconditioner.
        """
        self.event_shape = event_shape
        self.base_neg_log_prob_target = neg_log_prob_target

        self._n_steps: int = 0
        self._n_calls: int = 0  # Target density evaluation counter
        self._n_grads: int = 0  # Target density gradient evaluation counter
        # Counts the number of chains that diverged across all steps
        self._n_divergences: int = 0
        self._n_divergences_per_chain: torch.Tensor = 0

        if preconditioner is None:
            preconditioner = IdentityPreconditioner(event_shape)
        self._preconditioner = preconditioner

        # Used to determine warmup behavior in kernel implementations
        self._warmup_flag = False
        self._n_warmup_chains: int = None

    @property
    def warmup_active(self):
        return self._warmup_flag

    def start_warmup(self, n_chains: int):
        self._warmup_flag = True
        self._n_warmup_chains = n_chains

    def end_warmup(self):
        self._warmup_flag = False

    def reset_parameters(self):
        """
        Resets the parameters of this kernel to their default values.
        To be used within `MarkovKernel.warmup(...)`.
        """
        raise NotImplementedError

    def finalize_parameters(self):
        """
        Finalizes the parameters of this kernel after performing warmup.
        Default: do nothing.
        """
        pass

    def neg_log_prob_target(self, z: torch.Tensor):
        """
        Returns the negative log probability density of the preconditioner-adjusted target distribution.

        :param torch.Tensor z: latent tensor with shape `(*batch_shape, *event_shape)`.
        :return: negative log probability tensor with shape `batch_shape`.
        """
        x, log_det_inverse = self._preconditioner.inverse_transform(z)
        return self.base_neg_log_prob_target(x) - log_det_inverse

    def fit_preconditioner(self, x_train, **kwargs):
        self._preconditioner.fit(
            x=x_train,
            reset_optimizer=False,
            **kwargs
        )

    def calls_per_second(self, elapsed_time_seconds):
        if elapsed_time_seconds > 0:
            return self._n_calls / elapsed_time_seconds
        return torch.nan

    def grads_per_second(self, elapsed_time_seconds):
        if elapsed_time_seconds > 0:
            return self._n_grads / elapsed_time_seconds
        return torch.nan

    def pbar_repr(self, elapsed_time_seconds: float):
        """
        Returns a string that represents this object in sampling/warmup progress bars.
        """
        data = [
            self.name,
            f'{self.calls_per_second(elapsed_time_seconds):.3f} c/s',
            f'{self.grads_per_second(elapsed_time_seconds):.3f} g/s',
        ]
        return ', '.join(data)

    def set_target(self, new_neg_log_prob_target: callable):
        self.neg_log_prob_target = new_neg_log_prob_target

    @property
    def event_size(self):
        return int(torch.prod(torch.as_tensor(self.event_shape)))

    @property
    def name(self) -> str:
        raise NotImplementedError

    def increment_n_steps(self):
        self._n_steps += 1

    def increment_n_calls(self, n_calls: int):
        self._n_calls += n_calls
        self._n_calls = int(self._n_calls)

    def increment_n_grads(self, n_grads: int):
        self._n_grads += n_grads
        self._n_grads = int(self._n_grads)

    def increment_n_divergences(self, n_divergences: int):
        self._n_divergences += n_divergences
        self._n_divergences = int(self._n_divergences)
    
    def increment_n_divergences_per_chain(self, divergence_mask: torch.Tensor):
        self._n_divergences_per_chain += divergence_mask.long()
    
    def step_with_preconditioner_inverse(self, *args, **kwargs):
        z = self.step(*args, **kwargs)
        x = self._preconditioner.inverse_transform(z)[0]
        return z, x

    def step(self,
             x: torch.Tensor,
             *args,
             update: bool = False,
             **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs one kernel transition.

        :param torch.Tensor x: current state tensor with shape `(*batch_shape, *event_shape)`.
        :param bool update: if True, update kernel parameters.
        :param bool return_preconditioner_inverse: if True, return an additional tensor with shape 
         `(*batch_shape, *event_shape)`, which is after applying the preconditioner inverse.
        :return: new state tensor with shape `(*batch_shape, *event_shape)`.
        """
        raise NotImplementedError

    def reset_statistics(self):
        self._n_steps = 0
        self._n_calls = 0
        self._n_grads = 0
        self._n_divergences = 0


class CompositionKernel(MarkovKernel):
    """
    Composition of two or more Markov kernels.

    This means applying Markov kernels one after another in a Markov chain.
    Note: All kernels must leave the target distribution invariant. A warning is shown if the kernels use a different 
     target log probability density callable object (checked by object identity).
    """

    def __init__(self,
                 kernels: List[MarkovKernel],
                 mode: str = 'full',
                 schedule: List[int] = None):
        """
        CompositionKernel constructor.

        :param List[MarkovKernel] kernels: list of Markov kernel objects.
        :param str mode: one of ['full', 'cyclic']. If 'full', applies all kernels in a single step. If 'cyclic', 
         applies only the next kernel in a single step.
        :param List[int] schedule: list with one int for each kernel. If schedule[i] == k, then the kernels[i] is 
         applied k times in 'cyclic' mode. If None, each kernel is only applied once. Does not work in 'full' mode.
        """
        if len(kernels) < 1:
            raise ValueError("At least one kernel must be provided.")
        if schedule is None:
            schedule = [1] * len(kernels)
        else:
            if mode != 'cyclic':
                raise ValueError(
                    "Kernel schedule is only supported in cyclic mode.")
            if len(schedule) != len(kernels):
                raise ValueError(
                    "schedule must be None or have the same length as kernels.")
            for i in range(len(schedule)):
                if not isinstance(schedule[i], int):
                    raise ValueError("All schedule elements must be integers")
        for k in kernels[1:]:
            if not (k.event_shape is kernels[0].event_shape):
                raise ValueError(
                    "All kernels must have the same event shape."
                )
            if not (k.neg_log_prob_target is kernels[0].neg_log_prob_target):
                warnings.warn(
                    f'Kernel {k} uses different negative log probability density callable than kernel {kernels[0]}. '
                    f'Ensure that the target distribution is left invariant!'
                )
        if mode not in ['full', 'cyclic']:
            raise ValueError("mode must be one of ['full', 'cyclic']")

        super().__init__(
            event_shape=kernels[0].event_shape,
            neg_log_prob_target=kernels[0].neg_log_prob_target
        )
        self.schedule = schedule
        self._schedule_cumsum = np.cumsum(schedule)
        self._schedule_total = sum(self.schedule)

        self.kernels = kernels
        self.schedule_index = 0
        self.mode = mode

    def start_warmup(self, n_chains):
        for k in self.kernels:
            k.start_warmup(n_chains)
        super().start_warmup(n_chains)

    def reset_parameters(self):
        for k in self.kernels:
            k.reset_parameters()

    def set_target(self, new_neg_log_prob_target: callable):
        for k in self.kernels:
            k.neg_log_prob_target = new_neg_log_prob_target

    def get_cyclic_kernel_index(self):
        for i, val in enumerate(self._schedule_cumsum):
            if self.schedule_index < val:
                return i
        raise RuntimeError("Error retrieving kernel index")

    def step(self,
             x: torch.Tensor,
             update: bool = False,
             **kwargs):
        """
        Performs a transition with the next kernel if in cyclic mode.
        Performs a transition with all kernels if in full mode.

        :param torch.Tensor x: current state tensor with shape `(*batch_shape, *event_shape)`.
        :param bool update: if True, update this kernel's parameters.
        :return: new state tensor with shape `(*batch_shape, *event_shape)`.
        """

        if self.mode == 'cyclic':
            kernel_index = self.get_cyclic_kernel_index()
            x = self.kernels[kernel_index].step(x=x, update=update, **kwargs)
            self.schedule_index = (
                self.schedule_index + 1
            ) % self._schedule_total
            return x
        else:
            for k in self.kernels:
                x = k.step(x, update=update, **kwargs)
            return x

    def step_with_preconditioner_inverse(self,
                                         x: torch.Tensor,
                                         update: bool = False,
                                         **kwargs):
        if self.mode == 'cyclic':
            kernel_index = self.get_cyclic_kernel_index()
            z, x = self.kernels[kernel_index].step_with_preconditioner_inverse(
                x=x, update=update, **kwargs
            )
            self.schedule_index = (
                self.schedule_index + 1
            ) % self._schedule_total
            return z, x
        else:
            z = self.step(x=x, update=update, **kwargs)
            x = self.kernels[-1]._preconditioner.inverse_transform(z)[0]
            return z, x

    def reset_statistics(self):
        for k in self.kernels:
            k.reset_statistics()

    def fit_preconditioner(self, x_train, **kwargs):
        # Do not train the same preconditioner twice
        unique_preconditioners = list(
            set([k._preconditioner for k in self.kernels]))
        for p in unique_preconditioners:
            p.fit(x=x_train, **kwargs)


class MixingKernel(MarkovKernel):
    """
    Mixing of two or more Markov kernels.

    This means applying one Markov kernel at each step of a Markov chain. Each kernel has an associated selection 
    probability. These probabilities determine which kernel is chosen for the transition.
    Note: All kernels must leave the target distribution invariant. A warning is shown if the kernels use a different 
     target log probability density callable object (checked by object identity).
    """

    def __init__(self,
                 kernels: List[MarkovKernel],
                 selection_probabilities: List[float] = None):
        """
        MixingKernel constructor.

        :param List[MarkovKernel] kernels: list of Markov kernel objects.
        :param List[float] selection_probabilities: list of selection probabilities, one probability for each kernel.
         These probabilities must sum to 1. If None, set uniform selection probabilities.
        """
        if len(kernels) < 1:
            raise ValueError("At least one kernel must be provided.")
        for k in kernels[1:]:
            if not (k.event_shape is kernels[0].event_shape):
                raise ValueError(
                    "All kernels must have the same event shape."
                )
            if not (k.neg_log_prob_target is kernels[0].neg_log_prob_target):
                warnings.warn(
                    f'Kernel {k} uses different negative log probability density callable than kernel {kernels[0]}. '
                    f'Ensure that the target distribution is left invariant!'
                )

        super().__init__(
            event_shape=kernels[0].event_shape,
            neg_log_prob_target=kernels[0].neg_log_prob_target
        )
        self.kernels = kernels
        self.dist = None  # Categorical distribution for kernel selection

        if selection_probabilities is None:
            selection_probabilities = [
                1 / len(kernels) for _ in kernels
            ]

        # creates self.dist
        self.set_selection_probabilities(selection_probabilities)

    def start_warmup(self, n_chains):
        for k in self.kernels:
            k.start_warmup(n_chains)
        super().start_warmup(n_chains)

    @property
    def name(self) -> str:
        return f"Mix[{', '.join([k.name for k in self.kernels])}]"

    def pbar_repr(self, elapsed_time_seconds):
        return ", ".join([
            f"[{k.pbar_repr(elapsed_time_seconds)}]"
            for k in self.kernels
        ])

    def set_selection_probabilities(self, probs: List[float]):
        """
        Sets new selection probabilities.
        """
        if len(probs) != len(self.kernels):
            raise ValueError(
                "The number of kernels and selection probabilities must be the same.")
        if not math.isclose(sum(probs), 1.0):
            raise ValueError("Selection probabilities must sum to 1")
        self.dist = torch.distributions.Categorical(
            probs=torch.tensor(
                probs,
                dtype=torch.float
            )
        )

    def reset_parameters(self):
        """
        Resets the parameters of all kernels to their default values.
        Does not reset the selection probabilities.
        """
        for k in self.kernels:
            k.reset_parameters()

    def set_target(self, new_neg_log_prob_target: callable):
        for k in self.kernels:
            k.neg_log_prob_target = new_neg_log_prob_target

    def step(self,
             x: torch.Tensor,
             update: bool = False,
             **kwargs):
        """
        Performs a transition with a randomly chosen kernel.

        :param torch.Tensor x: current target-space state tensor with shape `(*batch_shape, *event_shape)`.
        :param bool update: if True, update this kernel's parameters.
        :return: new state tensor with shape `(*batch_shape, *event_shape)`.
        """
        idx = self.dist.sample()
        return self.kernels[idx].step(
            x=x,
            update=update,
            **kwargs
        )

    def step_with_preconditioner_inverse(self,
                                         x: torch.Tensor,
                                         update: bool = False,
                                         **kwargs):
        """
        Performs a transition with a randomly chosen kernel.
        Returns the preconditioner-inverse of the new state.

        :param torch.Tensor x: current target-space state tensor with shape `(*batch_shape, *event_shape)`.
        :param bool update: if True, update this kernel's parameters.
        :return: new state tensor with shape `(*batch_shape, *event_shape)`.
        """
        idx = self.dist.sample()
        return self.kernels[idx].step_with_preconditioner_inverse(
            x=x,
            update=update,
            **kwargs
        )

    def reset_statistics(self):
        for k in self.kernels:
            k.reset_statistics()

    def fit_preconditioner(self, x_train, **kwargs):
        # Do not train the same preconditioner twice
        unique_preconditioners = list(
            set([k._preconditioner for k in self.kernels]))
        for p in unique_preconditioners:
            p.fit(x=x_train, **kwargs)
