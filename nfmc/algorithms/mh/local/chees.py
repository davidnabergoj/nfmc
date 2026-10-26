import torch


class ChEES:
    """
    Class for tuning trajectory length and step size in HMC by maximizing the 
     change in the estimator of the expected square (ChEES).

    This class depends on scipy and numpy.
    """

    def __init__(self,
                 initial_step_size: torch.Tensor):
        """
        ChEES constructor.

        :param torch.Tensor initial_step_size: initial step size with shape `(n_chains,)`.
        """
        self.eps = initial_step_size  # Step size
        self.tlen = self.eps  # Trajectory length

        # Moving average of trajectory length
        self.tbar = torch.zeros_like(self.tlen)
        self.epsbar = torch.zeros_like(self.eps)  # Moving average of step size

        from scipy.stats.qmc import Halton
        import numpy as np

        self.halton = Halton(
            d=1,
            scramble=False,
            rng=np.random.default_rng(seed=0)
        )
        # Note: h.random(1) will deterministically draw the next number (and hence advance the sequence)

    def next_trajectory_length(self):
        h = torch.tensor(
            self.halton.random(1)[0, 0],
            dtype=self.tlen.dtype,
            device=self.tlen.device
        )
        return h * self.tlen

    def step(self, )