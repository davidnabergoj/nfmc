from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch

from nfmc.algorithms.util.expectation import MCMCExpectation, MCMCExpectationDict
from nfmc.algorithms.util.samples import MCMCSamples


@dataclass
class MHStatistics:
    event_shape: Union[Tuple[int, ...], torch.Size]
    n_accepted_trajectories: Optional[int] = 0
    n_attempted_trajectories: Optional[int] = 0
    n_divergences: Optional[int] = 0
    n_target_gradient_calls: Optional[int] = 0
    n_target_calls: Optional[int] = 0
    elapsed_time_seconds: Optional[float] = 0.0

    # transform data using this function when computing statistics
    data_transform: callable = lambda v: v
    expectations: MCMCExpectationDict = None

    def update_counters(self,
                        n_accepted_trajectories: int = 0,
                        n_attempted_trajectories: int = 0,
                        n_divergences: int = 0,
                        n_target_gradient_calls: int = 0,
                        n_target_calls: int = 0):
        self.n_accepted_trajectories = int(
            self.n_accepted_trajectories + n_accepted_trajectories)
        self.n_attempted_trajectories = int(
            self.n_attempted_trajectories + n_attempted_trajectories)
        self.n_divergences = int(self.n_divergences + n_divergences)
        self.n_target_gradient_calls = int(
            self.n_target_gradient_calls + n_target_gradient_calls)
        self.n_target_calls = int(self.n_target_calls + n_target_calls)

    def update_elapsed_time(self, delta_time_seconds: float):
        self.elapsed_time_seconds = float(
            self.elapsed_time_seconds + delta_time_seconds)

    def __post_init__(self):
        self.expectations = MCMCExpectationDict(
            {
                'first_moment': MCMCExpectation(self.event_shape, f=lambda v: v),
                'second_moment': MCMCExpectation(self.event_shape, f=lambda v: v ** 2),
            },
            data_transform=self.data_transform
        )

    @property
    def running_first_moment(self):
        return self.expectations['first_moment'].as_tensor()

    @property
    def running_second_moment(self):
        return self.expectations['second_moment'].as_tensor()

    @property
    def running_variance(self):
        return self.running_second_moment - self.running_first_moment ** 2

    @property
    def acceptance_rate(self):
        if self.n_attempted_trajectories == 0:
            return torch.nan
        return self.n_accepted_trajectories / self.n_attempted_trajectories

    @property
    def calls_per_second(self):
        if self.elapsed_time_seconds > 0:
            return self.n_target_calls / self.elapsed_time_seconds
        return torch.nan

    @property
    def grads_per_second(self):
        if self.elapsed_time_seconds > 0:
            return self.n_target_gradient_calls / self.elapsed_time_seconds
        return torch.nan

    def __repr__(self):
        return (
            f"acc-rate: {self.acceptance_rate:.2f}, "
            f"kcalls/s: {self.calls_per_second / 1000:.2f}, "
            f"kgrads/s: {self.grads_per_second / 1000:.2f}, "
            f"divergences: {self.n_divergences}"
        )

    def __dict__(self):
        return {
            'n_accepted_trajectories': self.n_accepted_trajectories,
            'n_attempted_trajectories': self.n_attempted_trajectories,
            'n_divergences': self.n_divergences,
            'n_target_gradient_calls': self.n_target_gradient_calls,
            'n_target_calls': self.n_target_calls,
            'elapsed_time_seconds': self.elapsed_time_seconds,
            'grads_per_second': self.grads_per_second,
            'acceptance_rate': self.acceptance_rate,
            'calls_per_second': self.calls_per_second,
        }


class MHOutput:
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 running_samples: MCMCSamples = None,
                 statistics: Optional[MHStatistics] = None,
                 store_samples: bool = True,
                 max_samples: int = None):
        self.event_shape = event_shape
        self.running_samples = running_samples
        self.statistics = statistics
        self.store_samples = store_samples
        self.max_samples = max_samples

        if self.running_samples is None:
            self.running_samples = MCMCSamples(
                self.event_shape,
                store_samples=self.store_samples,
                max_samples=self.max_samples,
            )
        if self.statistics is None:
            self.statistics = MHStatistics(self.event_shape)

    @property
    def samples(self) -> Union[torch.Tensor, None]:
        if not self.store_samples:
            return None
        return self.running_samples.as_tensor()

    def resample(self, n: int) -> torch.Tensor:
        flat = self.samples.flatten(0, 1)
        mask = torch.randint(low=0, high=len(flat), size=(n,))
        return flat[mask]  # (n, *event_shape)

    @property
    def mean(self):
        return self.statistics.running_first_moment

    @property
    def variance(self):
        return self.statistics.running_second_moment - self.statistics.running_first_moment ** 2

    @property
    def second_moment(self):
        return self.statistics.running_second_moment
