from typing import Tuple, Union, List
import math
import torch


class MarkovKernel:
    """
    Base MCMC kernel class.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 neg_log_prob_target: callable):
        """
        MarkovKernel constructor.

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param callable target: function that computes the negative log probability density of the target distribution.
         It receives as input a tensor with shape `(*batch_shape, *event_shape)` and outputs a tensor with shape
         `batch_shape`.
        """
        self.event_shape = event_shape
        self.neg_log_prob_target = neg_log_prob_target

        self._n_steps: int = 0
        self._n_calls: int = 0  # Target density evaluation counter
        self._n_grads: int = 0  # Target density gradient evaluation counter
        # Counts the number of chains that diverged across all steps
        self._n_divergences: int = 0

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

    def step(self,
             x: torch.Tensor,
             *args,
             update: bool = False,
             **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs one kernel transition.

        :param torch.Tensor x: current state tensor with shape `(*batch_shape, *event_shape)`.
        :param bool update: if True, update kernel parameters.
        :return: new state tensor with shape `(*batch_shape, *event_shape)`.
        """
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


class CompositionKernel(MarkovKernel):
    """
    Composition of two or more Markov kernels.

    This means applying Markov kernels one after another in a Markov chain.
    Note: All kernels must use the same target log probability density callable object (checked by object identity).
    """

    def __init__(self,
                 kernels: List[MarkovKernel],
                 mode: str = 'full'):
        """
        CompositionKernel constructor.

        :param List[MarkovKernel] kernels: list of Markov kernel objects.
        :param str mode: one of ['full', 'cyclic']. If 'full', applies all kernels in a single step. If 'cyclic', 
         applies only the next kernel in a single step.
        """
        if len(kernels) < 1:
            raise ValueError("At least one kernel must be provided.")
        for k in kernels[1:]:
            if not (k.event_shape is kernels[0].event_shape):
                raise ValueError(
                    "All kernels must have the same event shape."
                )
            if not (k.neg_log_prob_target is kernels[0].neg_log_prob_target):
                raise ValueError(
                    "All kernels must use the same negative log probability density callable."
                )
        if mode not in ['full', 'cyclic']:
            raise ValueError("mode must be one of ['full', 'cyclic']")

        super().__init__(
            event_shape=kernels[0].event_shape,
            neg_log_prob_target=kernels[0].neg_log_prob_target
        )
        self.kernels = kernels
        self.kernel_index = 0
        self.mode = mode

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
            out = self.kernels[self.kernel_index].step(
                x=x,
                update=update,
                **kwargs
            )
            self.kernel_index = (self.kernel_index + 1) % len(self.kernels)
            return out
        else:
            y = x
            for k in self.kernels:
                y = k.step(y, update=update, **kwargs)
            return y


class MixingKernel(MarkovKernel):
    """
    Mixing of two or more Markov kernels.

    This means applying one Markov kernel at each step of a Markov chain. Each kernel has an associated selection 
    probability. These probabilities determine which kernel is chosen for the transition.
    Note: all kernels must use the same target log probability density callable object (checked by object identity).
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
                raise ValueError(
                    "All kernels must use the same negative log probability density callable."
                )
        if selection_probabilities is None:
            selection_probabilities = [
                1 / len(kernels) for _ in kernels]
        else:
            if len(selection_probabilities) != len(kernels):
                raise ValueError(
                    "The number of kernels and selection probabilities must be the same.")
            if not math.isclose(sum(selection_probabilities), 1.0):
                raise ValueError("Selection probabilities must sum to 1")

        super().__init__(
            event_shape=kernels[0].event_shape,
            neg_log_prob_target=kernels[0].neg_log_prob_target
        )
        self.kernels = kernels
        self.dist = torch.distributions.Categorical(
            probs=torch.tensor(
                selection_probabilities,
                dtype=torch.float
            )
        )

    def step(self,
             x: torch.Tensor,
             update: bool = False,
             **kwargs):
        """
        Performs a transition with a randomly chosen kernel.

        :param torch.Tensor x: current state tensor with shape `(*batch_shape, *event_shape)`.
        :param bool update: if True, update this kernel's parameters.
        :return: new state tensor with shape `(*batch_shape, *event_shape)`.
        """
        idx = self.dist.sample()
        return self.kernels[idx].step(
            x=x,
            update=update,
            **kwargs
        )
