from typing import List, Tuple, Union

import torch


class MCMCSamples:
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 _running: List[torch.Tensor] = None,
                 last_sample: torch.Tensor = None,
                 thinning: int = 1,
                 seen_samples: int = 0):
        self.event_shape = event_shape
        self._running = _running
        self.last_sample = last_sample
        self.thinning = thinning
        self.seen_samples = seen_samples

    @property
    def store_samples(self):
        return self.max_samples > 0

    def __getitem__(self, index):
        if index == -1 or index == self.n_samples - 1:
            return self.last_sample
        return self._running[index]

    @property
    def n_samples(self) -> int:
        return len(self._running)

    def add(self, x: torch.Tensor):
        """
        Add x to running samples.

        :param x: tensor with shape `(n_chains, *event_shape)` or `(k, n_chains, *event_shape)`
        """
        if len(x) == 0:
            return

        # transform x into shape `(k, n_chains, *event_shape)`
        if len(x.shape) == len(self.event_shape) + 1 and x.shape[1:] == self.event_shape:
            x = x[None]
        elif len(x.shape) == len(self.event_shape) + 2 and x.shape[2:] == self.event_shape:
            pass
        else:
            raise ValueError(
                f"Expected x.shape[1:] or x.shape[2:] to be {self.event_shape}, got {x.shape = }")

        # Store the last sample
        self.last_sample = x[-1].detach().clone()

        if not self.store_samples:
            return

        if self.max_samples is None or self.n_samples + len(x) <= self.max_samples:
            thinning_mask = (torch.arange(self.seen_samples,
                             self.seen_samples + len(x)) % self.thinning) == 0
            self.seen_samples += len(x)
            added_samples = x[thinning_mask].detach().cpu()
            self._running.extend(added_samples)
        else:
            # Reservoir sampling
            for i in range(len(x)):
                if self.n_samples < self.max_samples:
                    self._running.append(x[i])
                else:
                    _idx = int(torch.randint(
                        low=0, high=self.n_samples, size=()).detach())
                    if _idx < self.max_samples:
                        self._running[_idx] = x[i]

    def as_tensor(self) -> torch.Tensor:
        if len(self._running) > 0:
            return torch.stack(self._running, dim=0)
        else:
            return torch.empty(size=(0, 0, *self.event_shape), dtype=torch.double)

    def reset(self):
        del self._running
        self._running = []
