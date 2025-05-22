import math


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

    def __repr__(self):
        return f'DA error: {self.error_sum:.2f}'
