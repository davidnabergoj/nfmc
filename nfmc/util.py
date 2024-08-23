from typing import Dict, List

import torch

FLOW_NAMES: Dict[str, List[str]] = {
    'realnvp': ["realnvp", 'real_nvp', 'rnvp'],
    'nice': ['nice'],
    'maf': ['maf'],
    'iaf': ['iaf'],
    'c-rqnsf': ['c-rqnsf'],
    'ma-rqnsf': ['ma-rqnsf', 'maf-rqnsf'],
    'ia-rqnsf': ['ia-rqnsf', 'iaf-rqnsf'],
    'c-lrsnsf': ['c-lrsnsf', 'c-lrs'],
    'ma-lrsnsf': ['ma-lrsnsf', 'maf-lrsnsf', 'ma-lrs', 'maf-lrs'],
    'ia-lrsnsf': ['ia-lrsnsf', 'iaf-lrsnsf', 'ia-lrs', 'iaf-lrs'],
    'ot-flow': ['ot-flow', 'otflow', 'ot flow'],
    'ffjord': ['ffjord'],
    'i-resnet': ['iresnet', 'invertible resnet', 'invertible-resnet', 'i-resnet'],
    'resflow': ['resflow', 'residual flow', 'residual-flow', 'res-flow'],
    'ddb': ['ddnf', 'ddb'],
    'c-naf-deep': ['c-naf-deep'],
    'c-naf-deep-dense': ['c-naf-deep-dense'],
    'c-naf-dense': ['c-naf-dense'],
    'proximal-resflow': ["proximal-resflow", 'p-resflow', 'presflow', 'proximal resflow'],
    'rnode': ["rnode", 'r-node'],
    'planar': ['planar'],
    'radial': ['radial'],
    'sylvester': ['sylvester'],
}


def is_flow_supported(flow_name: str):
    flow_name = flow_name.lower()
    for key, value in FLOW_NAMES.items():
        if flow_name in value:
            return True
    return False


def get_supported_normalizing_flows(synonyms: bool = True):
    supported = []
    for key, value in FLOW_NAMES.items():
        if synonyms:
            supported.extend(value)
        else:
            supported.append(key)
    return supported


def create_flow_object(flow_name: str, event_shape, **kwargs):
    assert is_flow_supported(flow_name)

    from torchflows.flows import Flow
    from torchflows.architectures import (
        RealNVP,
        MAF,
        IAF,
        CouplingRQNSF,
        MaskedAutoregressiveRQNSF,
        InverseAutoregressiveRQNSF,
        CouplingLRS,
        MaskedAutoregressiveLRS,
        InverseAutoregressiveLRS,
        CouplingDeepSF,
        CouplingDenseSF,
        CouplingDeepDenseSF,
        OTFlow,
        FFJORD,
        ResFlow,
        InvertibleResNet,
        DeepDiffeomorphicBijection,
        NICE,
        ProximalResFlow,
        RNODE,
        PlanarFlow,
        RadialFlow,
        SylvesterFlow
    )

    if flow_name in FLOW_NAMES['realnvp']:
        bijection = RealNVP(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES["nice"]:
        bijection = NICE(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES['maf']:
        bijection = MAF(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES['iaf']:
        bijection = IAF(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES['c-rqnsf']:
        bijection = CouplingRQNSF(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES['ma-rqnsf']:
        bijection = MaskedAutoregressiveRQNSF(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES['ia-rqnsf']:
        bijection = InverseAutoregressiveRQNSF(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES['c-lrsnsf']:
        bijection = CouplingLRS(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES['ma-lrsnsf']:
        bijection = MaskedAutoregressiveLRS(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES['ia-lrsnsf']:
        bijection = InverseAutoregressiveLRS(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES['ot-flow']:
        bijection = OTFlow(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES['ffjord']:
        bijection = FFJORD(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES['i-resnet']:
        bijection = InvertibleResNet(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES['resflow']:
        bijection = ResFlow(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES['ddb']:
        bijection = DeepDiffeomorphicBijection(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES["c-naf-deep"]:
        bijection = CouplingDeepSF(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES["c-naf-deep-dense"]:
        bijection = CouplingDeepDenseSF(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES["c-naf-dense"]:
        bijection = CouplingDenseSF(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES["proximal-resflow"]:
        bijection = ProximalResFlow(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES["rnode"]:
        bijection = RNODE(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES["planar"]:
        bijection = PlanarFlow(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES["radial"]:
        bijection = RadialFlow(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES["sylvester"]:
        bijection = SylvesterFlow(event_shape, **kwargs)
    else:
        raise ValueError

    return Flow(bijection)


def metropolis_acceptance_log_ratio(
        log_prob_curr,
        log_prob_prime,
        log_proposal_curr,
        log_proposal_prime
):
    # alpha = min(1, p(x_prime)/p(x_curr)*g(x_curr|x_prime)/g(x_prime|x_curr))
    # p = target
    # g(x_curr|x_prime) = log_proposal_curr
    # g(x_prime|x_curr) = log_proposal_prime
    return log_prob_prime - log_prob_curr + log_proposal_curr - log_proposal_prime


def compute_grad(fn_batched: callable, x):
    import torch
    with torch.enable_grad():
        x_clone = torch.clone(x)
        x_clone.requires_grad_(True)
        out = torch.autograd.grad(fn_batched(x_clone).sum(), x_clone)[0]
    out = out.detach()
    return out


def multivariate_normal_sample(batch_shape, event_shape, cov):
    """
    Draw samples from N(0, cov).
    If cov is None, we assume cov = identity.
    """
    if cov is None:
        samples = torch.randn(size=(*batch_shape, *event_shape))
    else:
        event_size = int(torch.prod(torch.as_tensor(event_shape)))
        assert cov.shape == (event_size, event_size)
        samples_dist = torch.distributions.MultivariateNormal(loc=torch.zeros(event_size), covariance_matrix=cov)
        samples_flat = samples_dist.sample(batch_shape)
        samples = samples_flat.view(*batch_shape, *event_shape)
    return samples
