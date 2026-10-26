import pytest
import torch
from nfmc.algorithms.preconditioning.preconditioners import (
    IdentityPreconditioner,
    DiagonalLinearPreconditioner,
    DenseLinearPreconditioner,
    NormalizingFlowPreconditioner,
)
from torchflows.flows import Flow
from torchflows.bijections.finite.autoregressive.architectures import RealNVP


EVENT_SHAPE = torch.Size([4])
BATCH_SIZE = 8
ATOL = 1e-5


def make_batch(event_shape=EVENT_SHAPE, batch_size=BATCH_SIZE):
    return torch.randn(batch_size, *event_shape)


def make_fitted_diag(x):
    p = DiagonalLinearPreconditioner(EVENT_SHAPE)
    p.fit(x)
    return p


def make_fitted_dense(x):
    p = DenseLinearPreconditioner(EVENT_SHAPE)
    p.fit(x)
    return p


def make_nfp():
    flow = Flow(RealNVP(EVENT_SHAPE))
    p = NormalizingFlowPreconditioner(flow)
    return p


@pytest.fixture
def x():
    torch.manual_seed(0)
    return make_batch()


@pytest.fixture
def z():
    torch.manual_seed(1)
    return make_batch() / 2


class TestIdentityPreconditionerAsDist:
    def test_forward_matches(self, x):
        p = IdentityPreconditioner(EVENT_SHAPE)
        dist = p.as_dist()

        x_p, log_det_p = p.forward_transform(x)
        x_d, log_det_d = dist.bijection.forward(x)

        assert torch.allclose(x_p, x_d, atol=ATOL)
        assert torch.allclose(log_det_p, log_det_d, atol=ATOL)

    def test_inverse_matches(self, z):
        p = IdentityPreconditioner(EVENT_SHAPE)
        dist = p.as_dist()

        x_p, log_det_p = p.inverse_transform(z)
        x_d, log_det_d = dist.bijection.inverse(z)

        assert torch.allclose(x_p, x_d, atol=ATOL)
        assert torch.allclose(log_det_p, log_det_d, atol=ATOL)


class TestDiagonalLinearPreconditionerAsDist:
    def test_forward_matches_unfitted(self, x):
        """Default (identity) parameters should agree with flow."""
        p = DiagonalLinearPreconditioner(EVENT_SHAPE)
        dist = p.as_dist()

        x_p, log_det_p = p.forward_transform(x)
        x_d, log_det_d = dist.bijection.forward(x)

        assert torch.allclose(x_p, x_d, atol=ATOL)
        assert torch.allclose(log_det_p, log_det_d, atol=ATOL)

    def test_inverse_matches_unfitted(self, z):
        p = DiagonalLinearPreconditioner(EVENT_SHAPE)
        dist = p.as_dist()

        x_p, log_det_p = p.inverse_transform(z)
        x_d, log_det_d = dist.bijection.inverse(z)

        assert torch.allclose(x_p, x_d, atol=ATOL)
        assert torch.allclose(log_det_p, log_det_d, atol=ATOL)

    def test_forward_matches_fitted(self, x):
        p = make_fitted_diag(x)
        dist = p.as_dist()

        x_p, log_det_p = p.forward_transform(x)
        x_d, log_det_d = dist.bijection.forward(x)

        assert torch.allclose(x_p, x_d, atol=ATOL)
        assert torch.allclose(log_det_p, log_det_d, atol=ATOL)

    def test_inverse_matches_fitted(self, x, z):
        p = make_fitted_diag(x)
        dist = p.as_dist()

        x_p, log_det_p = p.inverse_transform(z)
        x_d, log_det_d = dist.bijection.inverse(z)

        assert torch.allclose(x_p, x_d, atol=ATOL)
        assert torch.allclose(log_det_p, log_det_d, atol=ATOL)

    def test_forward_inverse_are_consistent(self, x):
        """forward_transform and inverse_transform should be inverses of each other."""
        p = make_fitted_diag(x)
        z, _ = p.forward_transform(x)
        x_rec, _ = p.inverse_transform(z)
        assert torch.allclose(x, x_rec, atol=ATOL)

    def test_log_dets_are_negatives(self, x):
        """Log-det of forward and inverse should sum to zero."""
        p = make_fitted_diag(x)
        _, ld_fwd = p.forward_transform(x)
        z, _ = p.forward_transform(x)
        _, ld_inv = p.inverse_transform(z)
        assert torch.allclose(
            ld_fwd + ld_inv, torch.zeros_like(ld_fwd), atol=ATOL)


class TestDenseLinearPreconditionerAsDist:
    def test_forward_matches_unfitted(self, x):
        p = DenseLinearPreconditioner(EVENT_SHAPE)
        dist = p.as_dist()

        x_p, log_det_p = p.forward_transform(x)
        x_d, log_det_d = dist.bijection.forward(x)

        assert torch.allclose(x_p, x_d, atol=ATOL)
        assert torch.allclose(log_det_p, log_det_d, atol=ATOL)

    def test_inverse_matches_unfitted(self, z):
        p = DenseLinearPreconditioner(EVENT_SHAPE)
        dist = p.as_dist()

        x_p, log_det_p = p.inverse_transform(z)
        x_d, log_det_d = dist.bijection.inverse(z)

        assert torch.allclose(x_p, x_d, atol=ATOL)
        assert torch.allclose(log_det_p, log_det_d, atol=ATOL)

    def test_forward_matches_fitted(self, x):
        p = make_fitted_dense(x)
        dist = p.as_dist()

        x_p, log_det_p = p.forward_transform(x)
        x_d, log_det_d = dist.bijection.forward(x)

        assert torch.allclose(x_p, x_d, atol=ATOL)
        assert torch.allclose(log_det_p, log_det_d, atol=ATOL)

    def test_inverse_matches_fitted(self, x, z):
        p = make_fitted_dense(x)
        dist = p.as_dist()

        x_p, log_det_p = p.inverse_transform(z)
        x_d, log_det_d = dist.bijection.inverse(z)

        assert torch.allclose(x_p, x_d, atol=ATOL)
        assert torch.allclose(log_det_p, log_det_d, atol=ATOL)

    def test_forward_inverse_are_consistent(self, x):
        p = make_fitted_dense(x)
        z, _ = p.forward_transform(x)
        x_rec, _ = p.inverse_transform(z)
        assert torch.allclose(x, x_rec, atol=ATOL)

    def test_log_dets_are_negatives(self, x):
        p = make_fitted_dense(x)
        _, ld_fwd = p.forward_transform(x)
        z, _ = p.forward_transform(x)
        _, ld_inv = p.inverse_transform(z)
        assert torch.allclose(
            ld_fwd + ld_inv, torch.zeros_like(ld_fwd), atol=ATOL)


class TestNormalizingFlowPreconditionerAsDist:
    def test_as_dist_returns_same_flow(self):
        """as_dist() should return the exact same Flow object."""
        p = make_nfp()
        assert p.as_dist() is p.flow

    def test_forward_matches(self, x):
        p = make_nfp()
        dist = p.as_dist()

        x_p, log_det_p = p.forward_transform(x)
        x_d, log_det_d = dist.bijection.forward(x)

        assert torch.allclose(x_p, x_d, atol=ATOL)
        assert torch.allclose(log_det_p, log_det_d, atol=ATOL)

    def test_inverse_matches(self, z):
        p = make_nfp()
        dist = p.as_dist()

        x_p, log_det_p = p.inverse_transform(z)
        x_d, log_det_d = dist.bijection.inverse(z)

        assert torch.allclose(x_p, x_d, atol=ATOL)
        assert torch.allclose(log_det_p, log_det_d, atol=ATOL)
