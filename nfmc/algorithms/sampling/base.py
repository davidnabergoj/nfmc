from dataclasses import dataclass
from typing import Optional, Any, Union, Tuple, List, Dict

import torch

from torchflows import Flow, RealNVP


@dataclass
class MCMCKernel:
    def __repr__(self):
        raise NotImplementedError

    def __post_init__(self):
        pass


@dataclass
class NFMCKernel(MCMCKernel):
    event_shape: Union[Tuple[int, ...], torch.Size]
    flow: Flow = None

    def __post_init__(self):
        super().__post_init__()
        if self.flow is None:
            self.flow = Flow(RealNVP(self.event_shape))


@dataclass
class MCMCParameters:
    n_iterations: int = 100
    n_warmup_iterations: int = 100
    tuning: bool = False
    store_samples: bool = True

    def __post_init__(self):
        pass

    def tuning_mode(self):
        self.tuning = True

    def sampling_mode(self):
        self.tuning = False


@dataclass
class NFMCParameters(MCMCParameters):
    train_pct: float = 0.7
    max_train_size: int = 4096
    max_val_size: int = 4096
    flow_fit_kwargs: dict[str, Any] = None

    def __post_init__(self):
        super().__post_init__()
        if self.flow_fit_kwargs is None:
            self.flow_fit_kwargs = {
                'early_stopping': True,
                'early_stopping_threshold': 50,
                'batch_size': 'adaptive',
                'show_progress': False
            }


@dataclass
class MCMCExpectation:
    """
    Compute E[f(x)] on streaming data.
    """

    event_shape: Union[torch.Size, Tuple[int, ...]]
    f: callable
    n_seen: int = 0
    running_value: Union[torch.Tensor, float] = 0.0  # shape: event_shape

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
        return self.running_value


class MCMCExpectationDict:
    def __init__(self, expectations: Dict[str, MCMCExpectation]):
        self.expectations = expectations

    def update(self, x: torch.Tensor):
        for k in self.expectations.keys():
            self.expectations[k].update(x)

    def reset(self):
        for k in self.expectations.keys():
            self.expectations[k].reset()

    def as_tensor(self):
        return {k: v.as_tensor() for k, v in self.expectations.items()}

    def __getitem__(self, key):
        return self.expectations[key]


@dataclass
class MCMCStatistics:
    event_shape: Union[Tuple[int, ...], torch.Size]
    n_accepted_trajectories: Optional[int] = 0
    n_attempted_trajectories: Optional[int] = 0
    n_divergences: Optional[int] = 0
    n_target_gradient_calls: Optional[int] = 0
    n_target_calls: Optional[int] = 0
    elapsed_time_seconds: Optional[float] = 0.0

    data_transform: callable = lambda v: v  # transform data using this function when computing statistics
    expectations: MCMCExpectationDict = None

    def __post_init__(self):
        self.expectations = MCMCExpectationDict({
            'first_moment': MCMCExpectation(self.event_shape, f=lambda v: self.data_transform(v)),
            'second_moment': MCMCExpectation(self.event_shape, f=lambda v: self.data_transform(v) ** 2),
        })

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


@dataclass
class MCMCSamples:
    event_shape: Union[Tuple[int, ...], torch.Size]
    store_samples: bool = True
    _running: List[torch.Tensor] = None  # shape: (n_iterations, n_chains, *event_shape)
    n_samples: int = 0
    last_sample: torch.Tensor = None  # shape (n_chains, *event_shape)
    thinning: int = 1
    seen_samples: int = 0

    def __post_init__(self):
        self.reset()

    def __getitem__(self, index):
        if index == -1 or index == self.n_samples - 1:
            return self.last_sample
        return self._running[index]

    def add(self, x: torch.Tensor):
        """
        Add x to running samples.

        :param x: tensor with shape `(n_chains, *event_shape)` or `(k, n_chains, *event_shape)`
        """
        # transform x into shape `(k, n_chains, *event_shape)`
        if len(x.shape) == len(self.event_shape) + 1 and x.shape[1:] == self.event_shape:
            x = x[None]
        elif len(x.shape) == len(self.event_shape) + 2 and x.shape[2:] == self.event_shape:
            pass
        else:
            raise ValueError(f"Expected x.shape[1:] or x.shape[2:] to be {self.event_shape}, got {x.shape = }")

        # Store the last sample
        self.last_sample = x[-1].detach().clone()

        if not self.store_samples:
            return

        thinning_mask = (torch.arange(self.seen_samples, self.seen_samples + len(x)) % self.thinning) == 0
        self.seen_samples += len(x)

        added_samples = x[thinning_mask].detach().cpu()
        self._running.extend(added_samples)
        self.n_samples += len(added_samples)

    def as_tensor(self) -> torch.Tensor:
        return torch.stack(self._running, dim=0)

    def reset(self):
        del self._running
        self._running = []
        self.n_samples = 0


@dataclass
class MCMCOutput:
    event_shape: Union[Tuple[int, ...], torch.Size]
    running_samples: MCMCSamples = None  # (n_iterations, n_chains, *event_shape)
    statistics: Optional[MCMCStatistics] = None
    kernel: Optional[MCMCKernel] = None
    store_samples: bool = True

    def __post_init__(self):
        if self.running_samples is None:
            self.running_samples = MCMCSamples(self.event_shape, store_samples=self.store_samples)
        if self.statistics is None:
            self.statistics = MCMCStatistics(self.event_shape)

    @property
    def samples(self) -> torch.Tensor:
        return self.running_samples.as_tensor()

    def resample(self, n: int) -> torch.Tensor:
        flat = self.samples.flatten(0, 1)
        mask = torch.randint(low=0, high=len(flat), size=(n,))
        return flat[mask]  # (n, *event_shape)

    def estimate_mean(self):
        return torch.mean(self.samples, dim=(0, 1))

    def estimate_variance(self):
        return torch.var(self.samples, dim=(0, 1))

    def estimate_second_moment(self):
        return self.estimate_variance() + self.estimate_mean() ** 2


class Sampler:
    """
    MCMC sampler class. Used for running MCMC/NFMC with fixed kernel and parameters.
    No burn-in or tuning phases are performed.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 target: callable,
                 kernel: MCMCKernel,
                 params: MCMCParameters):
        self.event_shape = event_shape
        self.target = target
        self.kernel = kernel
        self.params = params

    @property
    def name(self):
        return "Generic sampler"

    def warmup(self, x0: torch.Tensor, show_progress: bool = True, thinning: int = 1,
               time_limit_seconds: int = None) -> MCMCOutput:
        raise NotImplementedError

    def sample(self, x0: torch.Tensor, show_progress: bool = True, thinning: int = 1,
               time_limit_seconds: int = None) -> MCMCOutput:
        raise NotImplementedError
