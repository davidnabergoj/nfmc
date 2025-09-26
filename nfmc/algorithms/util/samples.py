from typing import List, Tuple, Union
import torch

from nfmc.algorithms.util.expectation import MCExpectation


class Samples:
    """
    Class that stores drawn samples.

    Samples are kept via reservoir sampling, i.e., when the number of samples exceeds a specified maximum, the reservoir
    ensures suitable replacement so that stored samples remain representative of the sampling procedure.
    Samples can be transformed via a user-defined functional.
    Stores the empirical first and second moment of samples.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 max_samples: int = None,
                 flatten: bool = False,
                 data_transform: callable = None):
        """
        Samples class constructor.

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param int max_samples: maximum number of samples to store in the reservoir. If None, all samples are stored.
        :param bool flatten: if True, flatten samples when calling `as_tensor()`, i.e., return a tensor of shape
         `(? , *event_shape)`. If False, return a tensor of shape `(n_steps, n_chains, *event_shape)`.
         If False and samples have different `n_chains`, calling `as_tensor()` raises a ValueError.
        :param callable data_transform: functional that transforms each added sample of type torch.Tensor with shape 
         `(*batch_shape, *event_shape)` into a torch.Tensor with shape `(*batch_shape, *event_shape)`. If None, no 
          transformation is applied.
        """
        self.event_shape = event_shape
        self.max_samples = max_samples
        self.flatten = flatten

        if data_transform is None:
            def data_transform(v): return v
        self.data_transform = data_transform

        self.first_moment = MCExpectation(event_shape)
        self.second_moment = MCExpectation(event_shape)

        self._last_sample: torch.Tensor = None
        self._running_samples: List[torch.Tensor] = []

    @property
    def last_sample(self) -> torch.Tensor:
        """
        Returns the last sample with shape `(n_chains, *event_shape)`.
        If there are no samples, returns `None`.
        """
        return self._last_sample

    @property
    def n_samples(self):
        return len(self._running_samples)

    def add(self, x: torch.Tensor) -> torch.Tensor:
        """
        Store sample x.

        :param torch.Tensor x: tensor with shape `(n_chains, *event_shape)` or `(k, n_chains, *event_shape)`
        :return: transformed tensor after applying self.data_transform. This tensor has the same shape as the input.
        """
        if len(x) == 0:
            return

        # Transform x into shape `(k, n_chains, *event_shape)`
        if len(x.shape) == len(self.event_shape) + 1 and x.shape[1:] == self.event_shape:
            x = x[None]
        elif len(x.shape) == len(self.event_shape) + 2 and x.shape[2:] == self.event_shape:
            pass
        else:
            raise ValueError(
                f"Expected x.shape[1:] or x.shape[2:] to be {self.event_shape}, got {x.shape = }"
            )

        # Apply data transform
        if self.data_transform is not None:
            x = self.data_transform(x)

        # Update first and second moments
        self.first_moment.update(x)
        self.second_moment.update(x ** 2)

        # Store the last sample separately
        self._last_sample = x[-1].detach()

        if self.max_samples is not None and self.max_samples < 1:
            return

        # Store samples inside a reservoir
        if self.max_samples is None or self.n_samples + len(x) <= self.max_samples:
            self._running_samples.extend(x.detach().cpu())
        else:
            # Reservoir sampling
            for i in range(len(x)):
                if self.n_samples < self.max_samples:
                    self._running_samples.append(x[i])
                else:
                    _idx = int(torch.randint(
                        low=0, high=self.n_samples, size=()).detach())
                    if _idx < self.max_samples:
                        self._running_samples[_idx] = x[i]
        return x

    def as_tensor(self) -> torch.Tensor:
        """
        Returns stored samples as a `torch.Tensor` with shape `(n_steps, n_chains, *event_shape)`.
        """
        if self.n_samples > 0:
            if self.flatten:
                return torch.concat(self._running_samples, dim=0)
            else:
                if len(set([len(e) for e in self._running_samples])) > 1:
                    raise ValueError(
                        "Samples have different n_chains, cannot return a tensor of shape "
                        "(n_steps, n_chains, *event_shape). Set flatten=True to concatenate samples along the first "
                        "dimension."
                    )
                return torch.stack(self._running_samples, dim=0)
        else:
            if self.flatten:
                return torch.empty(size=(0, *self.event_shape), dtype=torch.float)
            else:
                return torch.empty(size=(0, 0, *self.event_shape), dtype=torch.float)
