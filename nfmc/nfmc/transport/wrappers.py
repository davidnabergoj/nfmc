from base import Potential
from nfmc.nfmc.transport.stochastic_normalizing_flows import snf_hmc
from nfmc.nfmc.transport.nested import ns
from nfmc.nfmc.transport.deterministic_langevin import dlmc
from nfmc.nfmc.transport.annealed_flow_transport import aft, craft
from nfmc.nfmc.transport.flow_annealed_bootstrap import fab
from nfmc.util import create_flow_object


def aft_wrapper(target: Potential, prior: Potential, flow: str):
    flow_object = create_flow_object(flow)
    return aft(prior, target, flow_object, full_output=True)


def craft_wrapper(target: Potential, prior: Potential, flow: str):
    flow_object = create_flow_object(flow)
    return craft(prior, target, flow_object, full_output=True)


def dlmc_wrapper(target: Potential, prior: Potential, flow: str, n_particles: int = 100):
    x_prior = prior.sample(batch_shape=(n_particles,))
    flow_object = create_flow_object(flow)
    return dlmc(x_prior, lambda x: target(x) + prior(x), target, flow_object, full_output=True)


def fab_wrapper(target: Potential, flow: str):
    flow_object = create_flow_object(flow)
    return fab(target, flow_object)


def ns_wrapper(prior: Potential, target: Potential, flow: str, n_live: int = 100):
    flow_object = create_flow_object(flow)
    return ns(
        n_live_points=n_live,
        prior=prior,
        log_likelihood=lambda x: -target(x),
        flow=flow_object
    )


def snf_wrapper(prior: Potential, target: Potential, flow: str, n_particles: int = 100):
    return snf_hmc(
        prior_samples=prior.sample((n_particles,)),
        prior_potential=prior,
        target_potential=target,
        flow_name=flow
    )
