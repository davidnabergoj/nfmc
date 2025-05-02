from dataclasses import dataclass
from typing import Optional, Any, Union, Tuple, List, Dict

import torch

@dataclass
class NFMCKernel(MCMCKernel):
    flow: Any

    def __post_init__(self):
        from torchflows import Flow, RealNVP

        super().__post_init__()
        if self.flow is None:
            self.flow = Flow(RealNVP(self.event_shape))


@dataclass
class MCMCParameters:
    n_iterations: int = 100
    n_warmup_iterations: int = 100
    tuning: bool = False
    store_samples: bool = True
    max_samples: int = None

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



