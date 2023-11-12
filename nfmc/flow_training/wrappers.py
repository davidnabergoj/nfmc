from nfmc.flow_training import flow_annealed_importance_sampling_bootstrap_base
from potentials.base import Potential
from nfmc.util import create_flow_object


def fab(target: Potential, flow: str):
    """
    The prior distribution is the flow.
    FAB is more of a flow training method than a transport method.

    Possible enhancement: by using an identity flow initialization, we could let the user pass the prior potential,
    which would define the base flow distribution at the first iteration.
    """
    flow_object = create_flow_object(flow)
    return flow_annealed_importance_sampling_bootstrap_base(target, flow_object)
