from typing import Dict, Tuple, Union
import torch


class MCExpectation:
    """
    Compute E[f(x)] on streaming data using Monte Carlo.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 transform: callable = lambda v: v):
        self.event_shape = event_shape
        self.transform = transform
        self.running_value: Union[torch.Tensor, float] = 0.0
        self.n_seen: int = 0

    def update(self, x: torch.Tensor):
        """
        Update the running functional value.

        :param x: tensor with shape `(n_iterations, n_chains, *event_shape)` or `(n_chains, *event_shape)`.
        """
        if len(x.shape) == len(self.event_shape) + 2:
            pass
        elif len(x.shape) == len(self.event_shape) + 1:
            x = x[None]
        else:
            raise ValueError

        n_iterations, n_chains = x.shape[:2]
        n_new = n_iterations * n_chains

        self.running_value = torch.add(
            self.n_seen / (self.n_seen + n_new) * self.running_value,
            n_new / (self.n_seen + n_new) *
            torch.mean(self.transform(x.detach()).detach(), dim=(0, 1))
        )
        self.n_seen += n_new

    def as_tensor(self):
        return torch.as_tensor(self.running_value)


class MCExpectationDict:
    def __init__(self, expectations: Dict[str, MCExpectation], data_transform: callable):
        self.expectations = expectations
        self.data_transform = data_transform

    def update(self, x: torch.Tensor):
        if len(x) > 0:
            x_transformed = self.data_transform(x)
            for k in self.expectations.keys():
                self.expectations[k].update(x_transformed)

    def reset(self):
        for k in self.expectations.keys():
            self.expectations[k].reset()

    def as_tensor(self):
        return {k: v.as_tensor() for k, v in self.expectations.items()}

    def __getitem__(self, key):
        return self.expectations[key]
