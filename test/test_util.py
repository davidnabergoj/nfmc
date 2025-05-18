import torch
import pytest
from nfmc.util import grad_f
from test.util import standard_gaussian_neg_log_prob


@pytest.mark.parametrize('batch_shape', [(1,), (2,), (2, 3, 5)])
@pytest.mark.parametrize('event_shape', [(1,), (2,), (2, 3, 5)])
def test_grad_f(batch_shape, event_shape):
    torch.manual_seed(0)
    x = torch.randn(size=(*batch_shape, *event_shape))
    
    fval, gval, nc, ng = grad_f(x, standard_gaussian_neg_log_prob, event_shape)
    
    # torch.nan can be present
    assert isinstance(gval, torch.Tensor)
    assert torch.all(gval != torch.inf)
    assert fval.shape == batch_shape
    
    assert isinstance(fval, torch.Tensor)
    assert torch.all(fval != torch.inf)
    assert gval.shape == x.shape
    
    assert isinstance(nc, int)
    assert nc >= 0

    assert isinstance(ng, int)
    assert ng >= 0