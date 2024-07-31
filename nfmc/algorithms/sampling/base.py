from dataclasses import dataclass
from typing import Sized, Optional, Any

import torch

from normalizing_flows import Flow, RealNVP


@dataclass
class MCMCKernel:
    def __repr__(self):
        raise NotImplementedError

    def __post_init__(self):
        pass


@dataclass
class NFMCKernel(MCMCKernel):
    event_shape: Sized
    flow: Flow = None

    def __post_init__(self):
        super().__post_init__()
        if self.flow is None:
            self.flow = Flow(RealNVP(self.event_shape))


@dataclass
class MCMCParameters:
    n_iterations: int = 100

    def __post_init__(self):
        pass


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
                'batch_size': 'adaptive'
            }


@dataclass
class MCMCStatistics:
    n_accepted_trajectories: Optional[int] = 0
    n_attempted_trajectories: Optional[int] = 0
    n_divergences: Optional[int] = 0

    @property
    def acceptance_rate(self):
        if self.n_attempted_trajectories == 0:
            return torch.nan
        return self.n_accepted_trajectories / self.n_attempted_trajectories

    def __repr__(self):
        return (f"accepted: {self.n_accepted_trajectories} [{self.acceptance_rate:.3f}], "
                f"divergences: {self.n_divergences}")


@dataclass
class MCMCOutput:
    samples: torch.Tensor
    statistics: Optional[MCMCStatistics] = None


class Sampler:
    """
    MCMC sampler class. Used for running MCMC/NFMC with fixed kernel and parameters.
    No burn-in or tuning phases are performed.
    """

    def __init__(self,
                 event_shape: Sized,
                 target: callable,
                 kernel: MCMCKernel,
                 params: MCMCParameters):
        self.event_shape = event_shape
        self.target = target
        self.kernel = kernel
        self.params = params

    def sample(self, x0: torch.Tensor, show_progress: bool = True) -> MCMCOutput:
        raise NotImplementedError
