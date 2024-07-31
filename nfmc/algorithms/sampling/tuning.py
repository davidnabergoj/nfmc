import math
from dataclasses import dataclass


@dataclass
class DualAveragingParams:
    target_acceptance_rate: float = 0.651
    kappa: float = 0.75
    gamma: float = 0.05
    t0: int = 10


class DualAveraging:
    def __init__(self, initial_step_size, params: DualAveragingParams):
        self.t = params.t0
        self.error_sum = 0.0

        self.log_step_averaged = math.log(initial_step_size)
        self.log_step = math.inf
        self.mu = math.log(10 * initial_step_size)
        self.p = params

    def step(self, acceptance_rate_error):
        self.error_sum += float(acceptance_rate_error)  # This will eventually converge to 0 if all is well

        # Update raw step
        self.log_step = self.mu - self.error_sum / (math.sqrt(self.t) * self.p.gamma)

        # Update smoothed step
        eta = self.t ** -self.p.kappa
        self.log_step_averaged = eta * self.log_step + (1 - eta) * self.log_step_averaged
        self.t += 1

    @property
    def value(self):
        return math.exp(self.log_step_averaged)

    def __repr__(self):
        return f'DA error: {self.error_sum:.2f}'