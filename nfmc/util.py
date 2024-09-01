from typing import Dict, List

import torch

AFFINE_AUTOREGRESSIVE_FLOW_NAMES: Dict[str, List[str]] = {
    'realnvp': ["realnvp", 'real_nvp', 'rnvp'],
    'nice': ['nice'],
    'maf': ['maf'],
    'iaf': ['iaf'],
}

SPLINE_AUTOREGRESSIVE_FLOW_NAMES: Dict[str, List[str]] = {
    'c-rqnsf': ['c-rqnsf'],
    'ma-rqnsf': ['ma-rqnsf', 'maf-rqnsf'],
    'ia-rqnsf': ['ia-rqnsf', 'iaf-rqnsf'],
    'c-lrsnsf': ['c-lrsnsf', 'c-lrs'],
    'ma-lrsnsf': ['ma-lrsnsf', 'maf-lrsnsf', 'ma-lrs', 'maf-lrs'],
    'ia-lrsnsf': ['ia-lrsnsf', 'iaf-lrsnsf', 'ia-lrs', 'iaf-lrs'],
}

NEURAL_AUTOREGRESSIVE_FLOW_NAMES: Dict[str, List[str]] = {
    'c-naf-deep': ['c-naf-deep'],
    'ma-naf-deep': ['ma-naf-deep'],
    'ia-naf-deep': ['ia-naf-deep'],
    'c-naf-deep-dense': ['c-naf-deep-dense', 'c-naf-dense-deep'],
    'ma-naf-deep-dense': ['ma-naf-deep-dense', 'ma-naf-dense-deep'],
    'ia-naf-deep-dense': ['ia-naf-deep-dense', 'ia-naf-dense-deep'],
    'c-naf-dense': ['c-naf-dense'],
    'ma-naf-dense': ['ma-naf-dense'],
    'ia-naf-dense': ['ia-naf-dense'],
}

MULTISCALE_FLOW_NAMES: Dict[str, List[str]] = {
    'ms-realnvp': ['ms-realnvp', 'multiscale-realnvp'],
    'ms-nice': ['ms-nice', 'multiscale-nice'],
    'ms-rqnsf': ['ms-rqnsf', 'multiscale-rqnsf'],
    'ms-lrsnsf': ['ms-lrsnsf', 'multiscale-lrsnsf'],
    'ms-naf-deep': ['ms-naf-deep', 'multiscale-naf-deep'],
    'ms-naf-dense': ['ms-naf-dense', 'multiscale-naf-dense'],
    'ms-naf-deep-dense': ['ms-naf-deep-dense', 'multiscale-naf-deep-dense'],
    'glow': ['affine-glow', 'glow'],
}

AUTOREGRESSIVE_FLOW_NAMES: Dict[str, List[str]] = {
    **AFFINE_AUTOREGRESSIVE_FLOW_NAMES,
    **SPLINE_AUTOREGRESSIVE_FLOW_NAMES,
    **NEURAL_AUTOREGRESSIVE_FLOW_NAMES,
    **MULTISCALE_FLOW_NAMES,
}

CONTINUOUS_FLOW_NAMES: Dict[str, List[str]] = {
    'ot-flow': ['ot-flow', 'otflow', 'ot flow'],
    'ffjord': ['ffjord'],
    'ddb': ['ddnf', 'ddb'],
    'rnode': ["rnode", 'r-node'],
}

RESIDUAL_FLOW_NAMES: Dict[str, List[str]] = {
    'i-resnet': ['iresnet', 'invertible resnet', 'invertible-resnet', 'i-resnet'],
    'resflow': ['resflow', 'residual flow', 'residual-flow', 'res-flow'],
    'proximal-resflow': ["proximal-resflow", 'p-resflow', 'presflow', 'proximal resflow'],
    'planar': ['planar'],
    'radial': ['radial'],
    'sylvester': ['sylvester'],
}

FLOW_NAMES: Dict[str, List[str]] = {
    **AUTOREGRESSIVE_FLOW_NAMES,
    **CONTINUOUS_FLOW_NAMES,
    **RESIDUAL_FLOW_NAMES,
}


def flatten_name_dictionary(d: Dict[str, List[str]]) -> List[str]:
    flat = []
    flat.extend(list(d.keys()))
    for value in d.values():
        flat.extend(list(value))
    return sorted(list(set(flat)))


def is_flow_supported(flow_name: str):
    return flow_name in flatten_name_dictionary(FLOW_NAMES)


def get_supported_autoregressive_flows(synonyms: bool = True):
    if synonyms:
        return flatten_name_dictionary(AUTOREGRESSIVE_FLOW_NAMES)
    return sorted(list(AUTOREGRESSIVE_FLOW_NAMES.keys()))


def get_supported_residual_flows(synonyms: bool = True):
    if synonyms:
        return flatten_name_dictionary(RESIDUAL_FLOW_NAMES)
    return sorted(list(RESIDUAL_FLOW_NAMES.keys()))


def get_supported_continuous_flows(synonyms: bool = True):
    if synonyms:
        return flatten_name_dictionary(CONTINUOUS_FLOW_NAMES)
    return sorted(list(CONTINUOUS_FLOW_NAMES.keys()))


def get_supported_normalizing_flows(synonyms: bool = True):
    return sorted(list(set(
        get_supported_autoregressive_flows(synonyms=synonyms) +
        get_supported_residual_flows(synonyms=synonyms) +
        get_supported_continuous_flows(synonyms=synonyms)
    )))


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
        InverseAutoregressiveDeepSF,
        MaskedAutoregressiveDeepSF,
        CouplingDenseSF,
        InverseAutoregressiveDenseSF,
        MaskedAutoregressiveDenseSF,
        CouplingDeepDenseSF,
        InverseAutoregressiveDeepDenseSF,
        MaskedAutoregressiveDeepDenseSF,
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
        SylvesterFlow,
        MultiscaleLRSNSF,
        MultiscaleRQNSF,
        MultiscaleNICE,
        MultiscaleRealNVP,
        MultiscaleDeepSigmoid,
        MultiscaleDenseSigmoid,
        MultiscaleDeepDenseSigmoid,
        AffineGlow
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
    elif flow_name in FLOW_NAMES["ia-naf-deep"]:
        bijection = InverseAutoregressiveDeepSF(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES["ia-naf-deep-dense"]:
        bijection = InverseAutoregressiveDeepDenseSF(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES["ia-naf-dense"]:
        bijection = InverseAutoregressiveDenseSF(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES["ma-naf-deep"]:
        bijection = MaskedAutoregressiveDeepSF(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES["ma-naf-deep-dense"]:
        bijection = MaskedAutoregressiveDeepDenseSF(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES["ma-naf-dense"]:
        bijection = MaskedAutoregressiveDenseSF(event_shape, **kwargs)
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
    elif flow_name in FLOW_NAMES['ms-realnvp']:
        bijection = MultiscaleRealNVP(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES['ms-nice']:
        bijection = MultiscaleNICE(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES['ms-rqnsf']:
        bijection = MultiscaleRQNSF(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES['ms-lrsnsf']:
        bijection = MultiscaleLRSNSF(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES['ms-naf-deep']:
        bijection = MultiscaleDeepSigmoid(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES['ms-naf-deep-dense']:
        bijection = MultiscaleDeepDenseSigmoid(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES['ms-naf-dense']:
        bijection = MultiscaleDenseSigmoid(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES['glow']:
        bijection = AffineGlow(event_shape, **kwargs)
    else:
        raise ValueError

    return Flow(bijection)


def metropolis_acceptance_log_ratio(
        log_prob_target_curr,
        log_prob_target_prime,
        log_prob_proposal_curr,
        log_prob_proposal_prime
):
    # alpha = min(1, p(x_prime)/p(x_curr)*g(x_curr|x_prime)/g(x_prime|x_curr))
    # p = target
    # g(x_curr|x_prime) = log_proposal_curr
    # g(x_prime|x_curr) = log_proposal_prime
    return log_prob_target_prime - log_prob_target_curr + log_prob_proposal_curr - log_prob_proposal_prime


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
