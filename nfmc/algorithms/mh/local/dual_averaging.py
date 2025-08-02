import math
import torch


class DualAveraging:
    """
    Nesterov dual averaging class for Metropolis-Hastings step size tuning.
    Each chain's step size is tuned separately.

    The class requires space in the order of O(n_chains * n_steps) if storing errors or step sizes; and O(n_chains) if 
     not.
    """

    def __init__(self,
                 initial_step_size: float,
                 target_acceptance_rate: float,
                 kappa: float = 0.75,
                 gamma: float = 0.05,
                 t0: int = 10,
                 store_step_sizes: bool = False,
                 store_errors: bool = False):
        """
        DualAveraging constructor.

        :param float initial_step_size: initial positive step size for all chains.
        :param int n_chains: number of independent Metropolis-Hastings chains.
        :param float kappa:
        :param float gamma:
        :param int t0:
        :param bool store_step_sizes: if True, store step size after each step in a list.
        :param bool store_errors: if True, store acceptance rate errors after each step in a list.
        """
        self.target_acceptance_rate = target_acceptance_rate

        initial_step_size = initial_step_size

        self.t = t0
        self.kappa = kappa
        self.gamma = gamma

        self.error_sum = 0.0
        self.log_step_averaged = math.log(initial_step_size)
        self.log_step = torch.inf
        self.mu = math.log(10 * initial_step_size)

        self._store_step_sizes = store_step_sizes
        self._step_size_history = []

        self._store_errors = store_errors
        self._error_history = []

    @property
    def step_size_history(self):
        """
        Return step size history tensor with shape `(n_steps, n_chains)`.
        """
        if len(self._step_size_history) == 0:
            return self._step_size_history
        return torch.stack(self._step_size_history)

    @property
    def error_history(self):
        """
        Return error history tensor with shape `(n_steps, n_chains)`.
        """
        if len(self._error_history) == 0:
            return self._error_history
        return torch.stack(self._error_history)

    def step(self, acceptance_mask: torch.Tensor):
        """
        Update step size based on an incoming acceptance mask.

        :param torch.Tensor acceptance_mask: boolean tensor with shape `(n_chains,)` where the i-th element is true if
         the kernel transition corresponding to chain i was accepted.
        """
        if not isinstance(acceptance_mask, torch.Tensor):
            raise ValueError(
                f"Acceptance mask must be a tensor but got {type(acceptance_mask)}"
            )
        if len(acceptance_mask.shape) != 1:
            raise ValueError(
                f"Incorrect acceptance mask shape: {acceptance_mask.shape}"
            )

        acceptance_rate_error = self.target_acceptance_rate - acceptance_mask.float()

        # This will eventually converge to 0 if all is well
        self.error_sum += acceptance_rate_error

        # Update raw step
        self.log_step = (
            (self.mu - self.error_sum) / (math.sqrt(self.t) * self.gamma)
        )

        # Update smoothed step
        eta = self.t ** -self.kappa
        self.log_step_averaged = (
            (eta * self.log_step)
            + (1 - eta) * self.log_step_averaged
        )
        self.t += 1

        if self._store_step_sizes:
            self._step_size_history.append(self.value)
        if self._store_errors:
            self._error_history.append(acceptance_rate_error)

    @property
    def value(self):
        """
        Return step size tensor with shape `(n_chains,)`.
        """
        return torch.exp(torch.as_tensor(self.log_step_averaged))

    def weighted_value(self, sigma: float = 1.0):
        """
        Compute a weighted average of step sizes based on closeness to the target acceptance rate.

        :param sigma: controls sensitivity to deviation from target.
        :return: weighted final step size tensor with shape `(n_chains,)`.
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
        _errs = self.error_history.float()
        _log_steps = torch.log(self.step_size_history.float())

        _ws = torch.exp(-(_errs ** 2) / (2 * sigma ** 2))
        _ws = torch.exp(-(_errs ** 2) / (2 * sigma ** 2))
        _w_sum = torch.sum(_ws, dim=0)

        # Compute weighted step
        _w_log_step = torch.sum(_ws * _log_steps, dim=0) / _w_sum
        _w_log_step[_w_sum == 0] = _log_steps[-1][_w_sum == 0]  # Use last log step size if weights are zero for a chain

        _w_step = torch.exp(_w_log_step)
        return _w_step

    def __repr__(self):
        return f'DA error: {torch.mean(self.error_sum):.2f}'
