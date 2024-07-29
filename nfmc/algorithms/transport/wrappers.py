from potentials.base import Potential
from nfmc.transport_algorithms.stochastic_normalizing_flows import stochastic_normalizing_flow_hmc_base
from nfmc.transport_algorithms.nested import nested_sampling_base
from nfmc.transport_algorithms.annealed_flow_transport import annealed_flow_transport_base, \
    continual_repeated_annealed_flow_transport_base
from nfmc.util import create_flow_object


def aft(prior: Potential,
        target: Potential,
        flow: str,
        n_particles: int = 100,
        show_progress: bool = True,
        n_iterations: int = 20,
        **kwargs):
    flow_object = create_flow_object(flow, prior.event_shape)
    return annealed_flow_transport_base(
        prior,
        target,
        flow_object,
        n_particles=n_particles,
        n_steps=n_iterations,
        show_progress=show_progress,
        full_output=True,
        **kwargs
    )


def craft(prior: Potential,
          target: Potential,
          flow: str,
          n_particles: int = 100,
          n_iterations: int = 100,
          n_annealing_steps: int = 20,
          show_progress: bool = True,
          **kwargs):
    bijections = [
        create_flow_object(flow, prior.event_shape).bijection
        for _ in range(n_annealing_steps)
    ]
    return continual_repeated_annealed_flow_transport_base(
        prior,
        target,
        bijections,
        n_training_steps=n_iterations,
        n_annealing_steps=n_annealing_steps,
        n_particles=n_particles,
        show_progress=show_progress,
        **kwargs
    )


def ns(prior: Potential, target: Potential, flow: str, n_particles: int = 100):
    flow_object = create_flow_object(flow, prior.event_shape)
    return nested_sampling_base(
        n_live_points=n_particles,
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
