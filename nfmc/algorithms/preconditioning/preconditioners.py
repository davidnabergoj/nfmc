from typing import Tuple, Union
import torch
import torch.nn as nn
from nfmc.util import diag_mult, flatten_event
from torchflows.flows import Flow


class Preconditioner(nn.Module):
    """
    MCMC preconditioning class.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__()
        self.event_shape = event_shape

    def inverse_transform(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the inverse of z under this preconditioner.

        :param torch.Tensor z: latent tensor with shape `(*batch_shape, *event_shape)`.
        :return: tuple where the first element is the transformed latent tensor with shape 
         `(*batch_shape, *event_shape)` and the second element is the log of the absolute value of the Jacobian 
         determinant of this inverse transformation with respect to the latent tensor with shape `batch_shape`.
        """
        raise NotImplementedError

    def fit(self, x: torch.Tensor, **kwargs):
        """
        Update parameters of this preconditioner.

        :param torch.Tensor x: tensor of samples with shape `(*batch_shape, *event_shape)`. Note: these should
         be samples from the target space, not the latent space.
        :param kwargs:
        """
        raise NotImplementedError


class IdentityPreconditioner(Preconditioner):
    """
    Applies no preconditioning.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__(event_shape=event_shape)

    def inverse_transform(self, z: torch.Tensor):
        return z, torch.zeros(size=(z.shape[:-len(self.event_shape)])).to(z)

    def fit(self, x: torch.Tensor, **kwargs):
        pass


class DiagonalLinearPreconditioner(Preconditioner):
    """
    Applies diagonal linear preconditioning via `x = diag(v) @ z + loc`, where:
    - v is a vector with positive scalars,
    - loc is a real-valued location vector.

    When fitting, v becomes the training data standard deviation plus a small constant.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__(event_shape=event_shape)
        self.register_buffer('loc', torch.zeros(size=self.event_shape))
        self.register_buffer('v', torch.ones(size=self.event_shape))

    def inverse_transform(self, z: torch.Tensor):
        batch_shape = z.shape[:-len(self.event_shape)]
        fv = torch.log(self.v).sum()
        log_det = torch.full(size=batch_shape, fill_value=fv).to(z)
        return diag_mult(z, self.v, self.event_shape) + self.loc, log_det

    def fit(self, x: torch.Tensor, **kwargs):
        n_batch_dims = len(x.shape) - len(self.event_shape)
        batch_dims = list(range(n_batch_dims))
        self.v = torch.std(x, dim=batch_dims) + 1e-8
        self.loc = torch.mean(x, dim=batch_dims)


class DenseLinearPreconditioner(Preconditioner):
    """
    Applies dense linear preconditioning via `x = L @ z + loc`, where:
    - L is a lower-triangular positive definite matrix,
    - loc is a real-valued location vector.

    When fitting, L @ L.T = M, where M is the training data covariance plus a scaled identity matrix.
    In other words, M = Cov(z) + epsilon * I.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__(event_shape=event_shape)
        self.register_buffer('loc', torch.zeros(size=self.event_shape))
        self.register_buffer('tril_mat', torch.eye(self.event_size))

    @property
    def event_size(self):
        return int(torch.prod(torch.as_tensor(self.event_shape)))

    def inverse_transform(self, z: torch.Tensor):
        batch_shape = z.shape[:-len(self.event_shape)]
        z_flat = flatten_event(z, self.event_shape)
        x_flat = z_flat @ self.tril_mat.T  # L is lower-triangular
        x = x_flat.view_as(z) + self.loc
        fv = torch.log(torch.diag(self.tril_mat)).sum()
        log_det = torch.full(size=batch_shape, fill_value=fv).to(z)
        return x, log_det

    def fit(self, x: torch.Tensor, **kwargs):
        """
        :param torch.Tensor x: target space training data tensor with shape `(*batch_shape, *event_shape)`.
        """
        n_batch_dims = len(x.shape) - len(self.event_shape)
        batch_dims = list(range(n_batch_dims))
        x_flat = flatten_event(x, self.event_shape)
        cov = torch.cov(x_flat.T)  # (event_size, event_size)
        self.tril_mat = torch.linalg.cholesky(
            cov + 1e-8 * torch.eye(self.event_size)
        )
        self.loc = torch.mean(x, dim=batch_dims)


class NormalizingFlowPreconditioner(Preconditioner):
    def __init__(self,
                 flow: Flow):
        super().__init__(event_shape=flow.event_shape)
        self.flow: Flow = flow

    def inverse_transform(self, z: torch.Tensor):
        x, log_det = self.flow.bijection.inverse(z)
        return x, log_det

    def fit(self, x: torch.Tensor, **kwargs):
        """
        Does not use a train/validation split.
        """
        x_flat = x.view(-1, *self.event_shape)
        self.flow.fit(x_flat, **kwargs)
