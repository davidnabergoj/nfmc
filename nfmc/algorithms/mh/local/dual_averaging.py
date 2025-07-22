import math
import warnings
import numpy as np


class DualAveraging:
    """
    Nesterov dual averaging class for Metropolis-Hastings step size tuning.
    """

    def __init__(self,
                 initial_step_size: float,
                 kappa: float = 0.75,
                 gamma: float = 0.05,
                 t0: int = 10,
                 store_step_sizes: bool = False,
                 store_errors: bool = False):
        """
        DualAveraging constructor.

        :param float initial_step_size: initial positive step size.
        :param float kappa:
        :param float gamma:
        :param int t0:
        :param bool store_step_sizes: if True, store step size after each step in a list.
        :param bool store_errors: if True, store acceptance rate errors after each step in a list.
        """

        self.t = t0
        self.kappa = kappa
        self.gamma = gamma

        self.error_sum = 0.0
        self.log_step_averaged = math.log(initial_step_size)
        self.log_step = math.inf
        self.mu = math.log(10 * initial_step_size)

        self._store_step_sizes = store_step_sizes
        self._step_size_history = []
        self._store_errors = store_errors
        self._error_history = []

    @property
    def step_size_history(self):
        return self._step_size_history

    @property
    def error_history(self):
        return self._error_history

    def step(self, acceptance_rate_error):
        # This will eventually converge to 0 if all is well
        self.error_sum += float(acceptance_rate_error)

        # Update raw step
        self.log_step = self.mu - self.error_sum / \
            (math.sqrt(self.t) * self.gamma)

        # Update smoothed step
        eta = self.t ** -self.kappa
        self.log_step_averaged = eta * self.log_step + \
            (1 - eta) * self.log_step_averaged
        self.t += 1

        if self._store_step_sizes:
            self._step_size_history.append(self.value)
        if self._store_errors:
            self._error_history.append(acceptance_rate_error)

    @property
    def value(self):
        return math.exp(self.log_step_averaged)

    def weighted_value(self, sigma: float = 1.0):
        """
        Compute a weighted average of step sizes based on closeness to the target acceptance rate.

        :param sigma: controls sensitivity to deviation from target.
        :return: weighted final step size.
        """
        if len(self._error_history) == 0:
            raise ValueError(
                "Need error logging to compute weighted step"
            )
        if len(self._error_history) != len(self._step_size_history):
            raise ValueError(
                "Error and step histories must be of equal length"
            )

        # Prepare variables
        _errs = np.array(self._error_history).astype(np.float32)
        _log_steps = np.log(np.array(self._step_size_history).astype(np.float32))

        _ws = np.exp(-(_errs ** 2) / (2 * sigma ** 2))
        _ws = np.exp(-(_errs ** 2) / (2 * sigma ** 2))
        _w_sum = np.sum(_ws)

        # Return last step if weights are zero (probably should not happen)
        if _w_sum == 0:
            warnings.warn("Weights sum to zero, returning last step")
            return np.exp(_log_steps[-1])  # fallback

        # Compute weighted step
        _w_log_step = np.sum(_ws * _log_steps) / _w_sum
        return np.exp(_w_log_step)

    def __repr__(self):
        return f'DA error: {self.error_sum:.2f}'
