from typing import Dict, Tuple, Union
import torch


class MCMCExpectation:
    """
    Compute E[f(x)] on streaming data.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 f: callable):
        self.event_shape = event_shape
        self.f = f
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
            n_new / (self.n_seen + n_new) * torch.mean(self.f(x.detach()).detach(), dim=(0, 1))
        )
        self.n_seen += n_new

    def reset(self):
        self.n_seen = 0
        self.running_value = 0.0

    def as_tensor(self):
        return torch.as_tensor(self.running_value)
    
class MCMCExpectationDict:
    def __init__(self, expectations: Dict[str, MCMCExpectation], data_transform: callable):
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