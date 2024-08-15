import torch


def get_supported_normalizing_flows():
    return [
        "nice",
        "realnvp",
        "maf",
        # "iaf",
        "c-rqnsf",
        "ar-rqnsf",
        "c-lrsnsf",  # unstable
        "ar-lrsnsf",  # unstable
        "c-naf",
        # "ar-naf",
        # "c-bnaf",
        # "ar-bnaf",
        # "umnn-maf",
        # "planar",
        # "radial",
        # "sylvester",
        "i-resnet",  # needs 1 hour on cpu
        "resflow",  # needs 1 hour on cpu
        "proximal-resflow",
        "ffjord",  # needs 6 hours on cpu
        "rnode",
        "ddnf",
        "ot-flow",
    ]


def create_flow_object(flow_name: str, event_shape, **kwargs):
    from torchflows.flows import Flow
    from torchflows.architectures import (
        RealNVP,
        MAF,
        IAF,
        CouplingRQNSF,
        MaskedAutoregressiveRQNSF,
        CouplingLRS,
        MaskedAutoregressiveLRS,
        CouplingDSF,
        OTFlow,
        FFJORD,
        ResFlow,
        InvertibleResNet,
        DeepDiffeomorphicBijection,
        NICE,
        ProximalResFlow,
        RNODE
    )

    assert flow_name in get_supported_normalizing_flows()
    flow_name = flow_name.lower()

    if flow_name in ["realnvp"]:
        bijection = RealNVP(event_shape, **kwargs)
    elif flow_name in ["nice"]:
        bijection = NICE(event_shape, **kwargs)
    elif flow_name in ['maf']:
        bijection = MAF(event_shape, **kwargs)
    elif flow_name in ['iaf']:
        bijection = IAF(event_shape, **kwargs)
    elif flow_name in ['c-rqnsf']:
        bijection = CouplingRQNSF(event_shape, **kwargs)
    elif flow_name in ['ar-rqnsf']:
        bijection = MaskedAutoregressiveRQNSF(event_shape, **kwargs)
    elif flow_name in ['c-lrsnsf']:
        bijection = CouplingLRS(event_shape, **kwargs)
    elif flow_name in ['ar-lrsnsf']:
        bijection = MaskedAutoregressiveLRS(event_shape, **kwargs)
    elif flow_name in ['ot-flow', 'otflow']:
        bijection = OTFlow(event_shape, **kwargs)
    elif flow_name in ['ffjord']:
        bijection = FFJORD(event_shape, **kwargs)
    elif flow_name in ['iresnet', 'invertible resnet', 'invertible-resnet', 'i-resnet']:
        bijection = InvertibleResNet(event_shape, **kwargs)
    elif flow_name in ['resflow', 'residual flow', 'residual-flow', 'res-flow']:
        bijection = ResFlow(event_shape, **kwargs)
    elif flow_name in ['ddnf']:
        bijection = DeepDiffeomorphicBijection(event_shape, **kwargs)
    elif flow_name in ["c-naf"]:
        bijection = CouplingDSF(event_shape, **kwargs)
    elif flow_name in ["proximal-resflow"]:
        bijection = ProximalResFlow(event_shape, **kwargs)
    elif flow_name in ["rnode"]:
        bijection = RNODE(event_shape, **kwargs)
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
