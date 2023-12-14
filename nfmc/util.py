import math

import torch

# TODO add constants file to refer to NFs

from normalizing_flows import Flow
from normalizing_flows.bijections import (
    RealNVP,
    MAF,
    IAF,
    CouplingRQNSF,
    MaskedAutoregressiveRQNSF,
    CouplingLRS,
    MaskedAutoregressiveLRS,
    OTFlow,
    FFJORD,
    ResFlow,
    InvertibleResNet,
    DeepDiffeomorphicBijection,
    NICE
)


def get_supported_normalizing_flows():
    return [
        "nice",
        "realnvp",
        "maf",
        # "iaf",
        "c-rqnsf",
        "ar-rqnsf",
        "c-lrsnsf",
        "ar-lrsnsf",
        "c-naf",
        # "ar-naf",
        # "c-bnaf",
        # "ar-bnaf",
        # "umnn-maf",
        # "planar",
        # "radial",
        # "sylvester",
        # "i-resnet",
        # "resflow",
        # "proximal-resflow",
        # "ffjord",
        # "rnode",
        # "ddnf",
        # "ot-flow",
    ]


def create_flow_object(flow_name: str, event_shape, **kwargs):
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


class DualAveraging:
    def __init__(self, initial_value):
        self.h_sum = 0.
        self.x_bar = initial_value
        self.t = 0
        self.kappa = 0.75
        self.mu = math.log(10)
        self.gamma = 0.05
        self.t0 = 10

    def step(self, h_new):
        self.t += 1
        eta = self.t ** -self.kappa
        self.h_sum += h_new
        x_new = self.mu - math.sqrt(self.t) / self.gamma / (self.t + self.t0) * self.h_sum
        x_new_bar = eta * x_new + (1 - eta) * self.x_bar
        self.x_bar = x_new_bar

    @property
    def value(self):
        return self.x_bar


def compute_grad(fn_batched: callable, x: torch.Tensor):
    with torch.enable_grad():
        x_clone = torch.clone(x)
        x_clone.requires_grad_(True)
        out = torch.autograd.grad(fn_batched(x_clone).sum(), x_clone)[0]
    out = out.detach()
    return out


def relative_error(x_true, x_approx):
    return (x_true - x_approx) / x_true
