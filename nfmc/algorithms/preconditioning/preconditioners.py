from typing import Tuple, Union
import torch
import torch.nn as nn
from nfmc.util import diag_mult, flatten_event
from torchflows.flows import Flow
from torchflows.bijections.base import BijectiveComposition
from torchflows.bijections.finite.autoregressive.layers import ElementwiseScale, ElementwiseShift
from torchflows.bijections.finite.autoregressive.transformers.linear.affine import Scale


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

    def forward_transform(self, x: torch.Tensor):
        raise NotImplementedError

    def fit(self, x: torch.Tensor, **kwargs):
        """
        Update parameters of this preconditioner.

        :param torch.Tensor x: tensor of samples with shape `(*batch_shape, *event_shape)`. Note: these should
         be samples from the target space, not the latent space.
        :param kwargs:
        """
        raise NotImplementedError

    def as_dist(self) -> Flow:
        """Convert to distribution (via the normalizing flow class)."""
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

    def forward_transform(self, x: torch.Tensor):
        return x, torch.zeros(size=(x.shape[:-len(self.event_shape)])).to(x)

    def fit(self, x: torch.Tensor, **kwargs):
        pass

    def as_dist(self) -> Flow:
        from torchflows.bijections.finite.matrix import IdentityMatrix
        return Flow(IdentityMatrix(event_shape=self.event_shape))


class ElementwiseAffine(BijectiveComposition):
    def __init__(self, event_shape, context_shape=None, **kwargs):
        super().__init__([
            ElementwiseShift(event_shape=event_shape,
                             context_shape=context_shape, **kwargs),
            ElementwiseScale(event_shape=event_shape,
                             context_shape=context_shape, **kwargs),
        ])

    @torch.no_grad()
    def set_shift(self, shift: torch.Tensor):
        self.layers[0] = ElementwiseShift(
            event_shape=self.event_shape,
            fill_value=shift.reshape(self.layers[0].transformer.parameter_shape),
            context_shape=self.context_shape,
        )

    @torch.no_grad()
    def set_scale(self, scale: torch.Tensor):
        fill = Scale(torch.Size((1,))).unconstrain_alpha(scale)
        self.layers[1] = ElementwiseScale(
            event_shape=self.event_shape,
            fill_value=fill.reshape(self.layers[1].transformer.parameter_shape),
            context_shape=self.context_shape,
        )


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

    def forward_transform(self, x: torch.Tensor):
        batch_shape = x.shape[:-len(self.event_shape)]
        inv_v = 1.0 / self.v
        log_det = -torch.log(self.v).sum()
        log_det = torch.full(size=batch_shape, fill_value=log_det).to(x)
        z = diag_mult(x - self.loc, inv_v, self.event_shape)
        return z, log_det

    def fit(self, x: torch.Tensor, **kwargs):
        n_batch_dims = len(x.shape) - len(self.event_shape)
        batch_dims = list(range(n_batch_dims))
        self.v = torch.std(x, dim=batch_dims) + 1e-8
        self.loc = torch.mean(x, dim=batch_dims)

    def as_dist(self) -> Flow:
        bijection = ElementwiseAffine(event_shape=self.event_shape)
        bijection.set_shift(-self.loc)
        bijection.set_scale(1 / self.v)
        return Flow(bijection)


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

    def forward_transform(self, x: torch.Tensor):
        batch_shape = x.shape[:-len(self.event_shape)]
        x_centered = x - self.loc
        x_flat = flatten_event(x_centered, self.event_shape)
        z_flat = torch.linalg.solve_triangular(
            self.tril_mat, x_flat.T, upper=False).T
        z = z_flat.view_as(x)

        log_det = -torch.log(torch.diag(self.tril_mat)).sum()
        log_det = torch.full(size=batch_shape, fill_value=log_det).to(x)
        return z, log_det

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

    def as_dist(self) -> Flow:
        from torchflows.bijections.base import Bijection

        tril_mat = self.tril_mat
        loc = self.loc

        class DenseForwardTransformBijection(Bijection):
            """forward: z = L^-1 @ (x - loc)  [matches forward_transform]
               inverse: x = L @ z + loc        [matches inverse_transform]"""
            def __init__(self):
                super().__init__(event_shape=loc.shape)
                self.register_buffer('loc', loc)
                self.register_buffer('tril_mat', tril_mat)

            def forward(self, x, context=None):
                batch_shape = x.shape[:-len(self.event_shape)]
                x_flat = (x - self.loc).view(*batch_shape, -1)
                z_flat = torch.linalg.solve_triangular(
                    self.tril_mat, x_flat.T, upper=False
                ).T
                z = z_flat.view_as(x)
                log_det_val = -torch.log(torch.diag(self.tril_mat)).sum()
                log_det = torch.full(batch_shape, log_det_val.item()).to(x)
                return z, log_det

            def inverse(self, z, context=None):
                batch_shape = z.shape[:-len(self.event_shape)]
                z_flat = z.view(*batch_shape, -1)
                x_flat = (self.tril_mat @ z_flat.T).T
                x = x_flat.view_as(z) + self.loc
                log_det_val = torch.log(torch.diag(self.tril_mat)).sum()
                log_det = torch.full(batch_shape, log_det_val.item()).to(z)
                return x, log_det

        return Flow(DenseForwardTransformBijection())


class NormalizingFlowPreconditioner(Preconditioner):
    def __init__(self,
                 flow: Flow):
        super().__init__(event_shape=flow.event_shape)
        self.flow: Flow = flow

    def inverse_transform(self, z: torch.Tensor):
        self.flow.train()
        x, log_det = self.flow.bijection.inverse(z)
        self.flow.eval()
        return x, log_det

    def forward_transform(self, x: torch.Tensor):
        self.flow.train()
        z, log_det = self.flow.bijection.forward(x)
        self.flow.eval()
        return z, log_det

    def fit(self, x: torch.Tensor, lr: float = 1e-3, retries: int = 2, **kwargs):
        """
        Fit the flow to samples x from the target space.

        Does not use a train/validation split.

        :param torch.Tensor x: target space training data tensor with shape `(n_data, *event_shape)`.
        :param float lr: learning rate for flow training.
        :param int retries: number of retries with reduced learning rate if training fails.
        :param kwargs: additional keyword arguments passed to `self.flow.fit`.
        """
        x_flat = x.view(-1, *self.event_shape)
        try:
            self.flow.fit(x_flat, lr=lr, **kwargs)
        except RuntimeWarning as w:
            print(f"Flow training failed with warning: {w}.")
            print('Reducing learning rate')
            lr *= 0.1
            for _ in range(retries):
                try:
                    self.flow.fit(x_flat, lr=lr, **kwargs)
                    break
                except RuntimeWarning as w:
                    print(f"Flow training failed with warning: {w}.")
                    print('Reducing learning rate')
                    lr *= 0.1

    def as_dist(self) -> Flow:
        return self.flow
