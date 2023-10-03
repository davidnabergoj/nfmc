from potentials.base import Potential
from nfmc.transport.stochastic_normalizing_flows import stochastic_normalizing_flow_hmc_base
from nfmc.transport.nested import nested_sampling_base
from nfmc.transport.deterministic_langevin import deterministic_langevin_monte_carlo_base
from nfmc.transport.annealed_flow_transport import annealed_flow_transport_base, continual_repeated_annealed_flow_transport_base
from nfmc.util import create_flow_object


def aft(prior: Potential, target: Potential, flow: str):
    flow_object = create_flow_object(flow)
    return annealed_flow_transport_base(prior, target, flow_object, full_output=True)


def craft(prior: Potential, target: Potential, flow: str):
    flow_object = create_flow_object(flow)
    return continual_repeated_annealed_flow_transport_base(prior, target, flow_object, full_output=True)


def dlmc(prior: Potential, target: Potential, flow: str, n_particles: int = 100):
    x_prior = prior.sample(batch_shape=(n_particles,))
    flow_object = create_flow_object(flow)
    return deterministic_langevin_monte_carlo_base(x_prior, lambda x: target(x) + prior(x), target, flow_object, full_output=True)


def ns(prior: Potential, target: Potential, flow: str, n_live: int = 100):
    flow_object = create_flow_object(flow)
    return nested_sampling_base(
        n_live_points=n_live,
        prior=prior,
        log_likelihood=lambda x: -target(x),
        flow=flow_object
    )


def snf(prior: Potential, target: Potential, flow: str, n_particles: int = 100):
    return stochastic_normalizing_flow_hmc_base(
        prior_samples=prior.sample((n_particles,)),
        prior_potential=prior,
        target_potential=target,
        flow_name=flow
    )
