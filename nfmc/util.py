import json
from typing import Dict, List
import torch

FLOW_REFERENCE_DATA = {
    'realnvp': {'alt': ["realnvp", 'real_nvp', 'rnvp'], 'family': ('autoregressive', 'coupling', 'affine')},
    'ms-realnvp': {'alt': ['ms-realnvp', 'multiscale-realnvp'], 'family': ('autoregressive', 'multiscale', 'affine')},
    'glow-realnvp': {'alt': ['affine-glow', 'glow-affine', 'glow'],
                     'family': ('autoregressive', 'multiscale', 'affine')},
    'maf': {'alt': [], 'family': ('autoregressive', 'masked', 'affine')},
    'iaf': {'alt': [], 'family': ('autoregressive', 'masked', 'affine')},

    'nice': {'alt': [], 'family': ('autoregressive', 'coupling', 'affine')},
    'ms-nice': {'alt': ['ms-nice', 'multiscale-nice'], 'family': ('autoregressive', 'multiscale', 'affine')},
    'glow-nice': {'alt': ['shift-glow', 'glow-shift'], 'family': ('autoregressive', 'multiscale', 'affine')},

    'c-rqnsf': {'alt': ['c-rqsnsf'], 'family': ('autoregressive', 'coupling', 'spline')},
    'ms-rqnsf': {'alt': ['ms-rqnsf', 'multiscale-rqnsf'], 'family': ('autoregressive', 'multiscale', 'spline')},
    'glow-rqnsf': {'alt': ['rqs-glow', 'glow-rqs'], 'family': ('autoregressive', 'multiscale', 'spline')},
    'ma-rqnsf': {'alt': ['ma-rqsnsf', 'maf-rqsnsf', 'maf-rqnsf'], 'family': ('autoregressive', 'masked', 'spline')},
    'ia-rqnsf': {'alt': ['ia-rqsnsf', 'iaf-rqsnsf', 'iaf-rqnsf'], 'family': ('autoregressive', 'masked', 'spline')},

    'c-lrsnsf': {'alt': ['c-lrnsf'], 'family': ('autoregressive', 'coupling', 'spline')},
    'ms-lrsnsf': {'alt': ['ms-lrsnsf', 'multiscale-lrsnsf'], 'family': ('autoregressive', 'multiscale', 'spline')},
    'glow-lrsnsf': {'alt': ['lrs-glow', 'glow-lrs'], 'family': ('autoregressive', 'multiscale', 'spline')},
    'ma-lrsnsf': {'alt': ['ma-lrnsf', 'maf-lrsnsf', 'maf-lrnsf'], 'family': ('autoregressive', 'masked', 'spline')},
    'ia-lrsnsf': {'alt': ['ia-lrnsf', 'iaf-lrsnsf', 'iaf-lrnsf'], 'family': ('autoregressive', 'masked', 'spline')},

    'c-naf-deep': {'alt': [], 'family': ('autoregressive', 'coupling', 'nn')},
    'ms-naf-deep': {'alt': ['ms-naf-deep', 'multiscale-naf-deep'], 'family': ('autoregressive', 'multiscale', 'nn')},
    'glow-naf-deep': {'alt': ['naf-deep-glow', 'glow-naf-deep'], 'family': ('autoregressive', 'multiscale', 'nn')},
    'ma-naf-deep': {'alt': ['maf-naf-deep'], 'family': ('autoregressive', 'masked', 'nn')},
    'ia-naf-deep': {'alt': ['iaf-naf-deep'], 'family': ('autoregressive', 'masked', 'nn')},

    'c-naf-deep-dense': {'alt': [], 'family': ('autoregressive', 'coupling', 'nn')},
    'ms-naf-deep-dense': {'alt': ['ms-naf-deep-dense', 'multiscale-naf-deep-dense'],
                          'family': ('autoregressive', 'multiscale', 'nn')},
    'glow-naf-deep-dense': {'alt': ['naf-deep-dense-glow', 'glow-naf-deep-dense'],
                            'family': ('autoregressive', 'multiscale', 'nn')},
    'ma-naf-deep-dense': {'alt': ['maf-naf-deep-dense'], 'family': ('autoregressive', 'masked', 'nn')},
    'ia-naf-deep-dense': {'alt': ['iaf-naf-deep-dense'], 'family': ('autoregressive', 'masked', 'nn')},

    'c-naf-dense': {'alt': [], 'family': ('autoregressive', 'coupling', 'nn')},
    'ms-naf-dense': {'alt': ['ms-naf-dense', 'multiscale-naf-dense'], 'family': ('autoregressive', 'multiscale', 'nn')},
    'glow-naf-dense': {'alt': ['naf-dense-glow', 'glow-naf-dense'], 'family': ('autoregressive', 'multiscale', 'nn')},
    'ma-naf-dense': {'alt': ['maf-naf-dense'], 'family': ('autoregressive', 'masked', 'nn')},
    'ia-naf-dense': {'alt': ['iaf-naf-dense'], 'family': ('autoregressive', 'masked', 'nn')},

    'i-resnet': {'alt': ['iresnet', 'invertible resnet', 'invertible-resnet', 'i-resnet'],
                 'family': ('residual', 'iterative', 'standard')},
    'conv-i-resnet': {
        'alt': ['conv-iresnet', 'convolutional invertible resnet', 'conv-invertible-resnet', 'conv-i-resnet'],
        'family': ('residual', 'iterative', 'convolutional')},
    'resflow': {'alt': ['resflow', 'residual flow', 'residual-flow', 'res-flow'],
                'family': ('residual', 'iterative', 'standard')},
    'conv-resflow': {'alt': ['conv-resflow', 'convolutional residual flow', 'conv-residual-flow', 'conv-res-flow'],
                     'family': ('residual', 'iterative', 'convolutional')},
    'proximal-resflow': {'alt': ["proximal-resflow", 'p-resflow', 'presflow', 'proximal resflow'],
                         'family': ('residual', 'iterative', 'standard')},

    'planar': {'alt': [], 'family': ('residual', 'matrix-det')},
    'radial': {'alt': [], 'family': ('residual', 'matrix-det')},
    'sylvester': {'alt': [], 'family': ('residual', 'matrix-det')},

    'ot-flow': {'alt': ['ot-flow', 'otflow', 'ot flow'], 'family': ('continuous', 'standard')},
    'ffjord': {'alt': ['ffjord'], 'family': ('continuous', 'standard')},
    'conv-ffjord': {'alt': ['conv-ffjord'], 'family': ('continuous', 'convolutional')},
    'ddb': {'alt': ['ddnf'], 'family': ('continuous', 'standard')},
    'conv-ddb': {'alt': ['ddnf'], 'family': ('continuous', 'convolutional')},
    'rnode': {'alt': ['rnode'], 'family': ('continuous', 'standard')},
    'conv-rnode': {'alt': ['rnode'], 'family': ('continuous', 'convolutional')},
}


def get_flow_family(flow: str):
    try:
        return FLOW_REFERENCE_DATA[flow]['family']
    except KeyError:
        for key in FLOW_REFERENCE_DATA:
            if flow in FLOW_REFERENCE_DATA[key]['alt']:
                return FLOW_REFERENCE_DATA[key]['family']
    raise KeyError(f"Flow {flow} not found in reference data")


AFFINE_AUTOREGRESSIVE_FLOW_NAMES: Dict[str, List[str]] = {
    k: [k] + FLOW_REFERENCE_DATA[k]['alt'] for k in FLOW_REFERENCE_DATA.keys()
    if FLOW_REFERENCE_DATA[k]['family'][0] == 'autoregressive'
       and FLOW_REFERENCE_DATA[k]['family'][2] == 'affine'
       and FLOW_REFERENCE_DATA[k]['family'][1] in ['coupling', 'masked']
}

SPLINE_AUTOREGRESSIVE_FLOW_NAMES: Dict[str, List[str]] = {
    k: [k] + FLOW_REFERENCE_DATA[k]['alt'] for k in FLOW_REFERENCE_DATA.keys()
    if FLOW_REFERENCE_DATA[k]['family'][0] == 'autoregressive'
       and FLOW_REFERENCE_DATA[k]['family'][2] == 'spline'
       and FLOW_REFERENCE_DATA[k]['family'][1] in ['coupling', 'masked']
}

NEURAL_AUTOREGRESSIVE_FLOW_NAMES: Dict[str, List[str]] = {
    k: [k] + FLOW_REFERENCE_DATA[k]['alt'] for k in FLOW_REFERENCE_DATA.keys()
    if FLOW_REFERENCE_DATA[k]['family'][0] == 'autoregressive'
       and FLOW_REFERENCE_DATA[k]['family'][2] == 'nn'
       and FLOW_REFERENCE_DATA[k]['family'][1] in ['coupling', 'masked']
}

MULTISCALE_FLOW_NAMES: Dict[str, List[str]] = {
    k: [k] + FLOW_REFERENCE_DATA[k]['alt'] for k in FLOW_REFERENCE_DATA.keys()
    if FLOW_REFERENCE_DATA[k]['family'][0] == 'autoregressive'
       and FLOW_REFERENCE_DATA[k]['family'][1] == 'multiscale'
}

AUTOREGRESSIVE_FLOW_NAMES: Dict[str, List[str]] = {
    **AFFINE_AUTOREGRESSIVE_FLOW_NAMES,
    **SPLINE_AUTOREGRESSIVE_FLOW_NAMES,
    **NEURAL_AUTOREGRESSIVE_FLOW_NAMES,
    **MULTISCALE_FLOW_NAMES,
}

CONTINUOUS_FLOW_NAMES: Dict[str, List[str]] = {
    k: [k] + FLOW_REFERENCE_DATA[k]['alt'] for k in FLOW_REFERENCE_DATA.keys()
    if FLOW_REFERENCE_DATA[k]['family'][0] == 'continuous'
}

RESIDUAL_FLOW_NAMES: Dict[str, List[str]] = {
    k: [k] + FLOW_REFERENCE_DATA[k]['alt'] for k in FLOW_REFERENCE_DATA.keys()
    if FLOW_REFERENCE_DATA[k]['family'][0] == 'residual'
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


def parse_flow_string(flow_string: str):
    """
    Flow string syntax: <flow_name>%<json_string> or <flow_name>.
    """
    if flow_string is None:
        return {
            'name': None,
            'kwargs': {},
            'hash': hash('None')
        }

    if '%' not in flow_string:
        return {
            'name': flow_string,
            'kwargs': {},
            'hash': hash(flow_string)
        }
    else:
        flow_name = flow_string.split('%')[0]
        kwargs = json.loads(flow_string.split('%')[1])
        return {
            'name': flow_name,
            'kwargs': kwargs,
            'hash': hash(flow_name + str(kwargs))
        }


def create_flow_object(flow_string: str, event_shape, **kwargs):
    flow_data = parse_flow_string(flow_string)
    flow_name = flow_data['name']
    kwargs.update(flow_data['kwargs'])

    if isinstance(flow_name, str):
        assert is_flow_supported(flow_name)
    else:
        raise ValueError

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
        AffineGlow,
        ShiftGlow,
        RQSGlow,
        LRSGlow,
        DeepSigmoidGlow,
        DeepDenseSigmoidGlow,
        DenseSigmoidGlow,
        ConvolutionalRNODE,
        ConvolutionalFFJORD,
        ConvolutionalDeepDiffeomorphicBijection,
        ConvolutionalResFlow,
        ConvolutionalInvertibleResNet
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
    elif flow_name in FLOW_NAMES['glow-realnvp']:
        bijection = AffineGlow(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES['glow-nice']:
        bijection = ShiftGlow(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES['glow-rqnsf']:
        bijection = RQSGlow(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES['glow-lrsnsf']:
        bijection = LRSGlow(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES['glow-naf-deep']:
        bijection = DeepSigmoidGlow(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES['glow-naf-dense']:
        bijection = DenseSigmoidGlow(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES['glow-naf-deep-dense']:
        bijection = DeepDenseSigmoidGlow(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES['conv-i-resnet']:
        bijection = ConvolutionalInvertibleResNet(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES['conv-resflow']:
        bijection = ConvolutionalResFlow(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES['conv-ffjord']:
        bijection = ConvolutionalFFJORD(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES['conv-rnode']:
        bijection = ConvolutionalRNODE(event_shape, **kwargs)
    elif flow_name in FLOW_NAMES['conv-ddb']:
        bijection = ConvolutionalDeepDiffeomorphicBijection(event_shape, **kwargs)
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


def get_supported_mcmc_samplers() -> List[str]:
    return ['hmc', 'uhmc', 'ula', 'mala', 'mh', 'ess']


def get_supported_nfmc_samplers() -> List[str]:
    return [
        "imh",
        "jump_mala",
        "jump_ula",
        "jump_hmc",
        "jump_uhmc",
        "jump_ess",
        "jump_mh",
        "neutra_hmc",
        "tess",
        "dlmc"
    ]


def get_supported_samplers() -> List[str]:
    return get_supported_mcmc_samplers() + get_supported_nfmc_samplers()
