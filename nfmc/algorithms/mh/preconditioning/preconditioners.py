from typing import Tuple, Union
import torch
import torch.nn as nn
from nfmc.algorithms.mh.preconditioning.base import Preconditioner
from nfmc.util import diag_mult


class DiagonalLinearPreconditioner(Preconditioner):
    """
    Applies diagonal linear preconditioning via `x = diag(v) @ z` where v is a vector with positive scalars.

    When fitting, v becomes the training data standard deviation plus a small constant.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__(event_shape=event_shape)
        self.v = torch.ones(size=self.event_shape)

    def inverse_transform(self, z: torch.Tensor):
        batch_shape = z.shape[:-len(self.event_shape)]
        fv = torch.log(self.v).sum()
        log_det = torch.full(size=batch_shape, fill_value=fv).to(z)
        return diag_mult(z, 1 / self.v, self.event_shape), log_det

    def fit(self, z: torch.Tensor):
        n_batch_dims = len(z.shape) - len(self.event_shape)
        batch_dims = list(range(n_batch_dims))
        self.v = torch.std(z, dim=batch_dims) + 1e-8


class DenseLinearPreconditioner(Preconditioner):
    """
    Applies dense linear preconditioning via `x = L @ z` where L is an upper-triangular positive definite matrix.

    When fitting, L @ L.T = M, where M is the training data covariance plus a scaled identity matrix.
    In other words, M = Cov(z) + epsilon * I.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__(event_shape=event_shape)
        self.tril_mat = torch.eye(self.event_size)  # (event_size, event_size)

    @property
    def event_size(self):
        return int (torch.prod(torch.as_tensor(self.event_shape)))

    def flatten_event(self, z: torch.Tensor):
        """
        Converts tensor with shape `(*batch_shape, *event_shape)` into a tensor with shape `(batch_size, event_size)`.
        """
        n_batch_dims = len(z.shape) - len(self.event_shape)
        # (batch_size, *event_shape)
        z_flat_tmp = z.flatten(start_dim=0, end_dim=n_batch_dims - 1)
        z_flat = z_flat_tmp.flatten(start_dim=1)  # (batch_size, event_size)
        return z_flat

    def inverse_transform(self, z: torch.Tensor):
        batch_shape = z.shape[:-len(self.event_shape)]
        z_flat = self.flatten_event(z)
        fv = -torch.log(torch.diag(self.tril_mat)).sum()
        log_det = torch.full(size=batch_shape, fill_value=fv).to(z)
        x_flat = torch.linalg.solve_triangular(self.tril_mat, z_flat.T, upper=False).T
        x = x_flat.view_as(z)
        return x, log_det

    def fit(self, z: torch.Tensor):
        """
        :param torch.Tensor z: training data tensor with shape `(*batch_shape, *event_shape)`.
        """
        z_flat = self.flatten_event(z)
        cov = torch.cov(z_flat.T)  # (event_size, event_size)
        self.tril_mat = torch.linalg.cholesky(cov + 1e-8 * torch.eye(self.event_size))
