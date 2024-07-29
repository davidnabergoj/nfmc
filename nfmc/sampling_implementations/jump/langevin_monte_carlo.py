import torch

from nfmc.sampling_implementations.jump.base import JumpMCMC
from nfmc.util import MCMCOutput
from normalizing_flows import Flow
from nfmc.mcmc.langevin_algorithm import base as base_langevin


class NFLMC(JumpMCMC):
    def __init__(self,
                 target_potential: callable,
                 flow: Flow,
                 mcmc_kwargs: dict = None,
                 inv_mass_diag: torch.Tensor = None,
                 step_size: float = None,
                 **kwargs):
        self.inv_mass_diag = inv_mass_diag
        self.step_size = step_size
        if mcmc_kwargs is None:
            mcmc_kwargs = dict()
        self.mcmc_kwargs = mcmc_kwargs

        super().__init__(target_potential, flow, **kwargs)

    @property
    def name(self) -> str:
        return "NF-LMC"

    def sample_mcmc(self, x: torch.Tensor):
        output = base_langevin(
            x0=x,
            n_iterations=self.jump_period - 1,
            target=self.target_potential,
            **{**self.mcmc_kwargs, **{
                "inv_mass_diag": self.inv_mass_diag,
                "step_size": self.step_size}
               }  # Reuse old kernel parameters and have them overwrite whatever is in kwargs
        )  # (n_steps, n_chains, *event_shape) where n_steps = self.jump_period - 1
        self.inv_mass_diag = output.kernel["inv_mass_diag"]
        self.step_size = output.kernel["step_size"]
        return output.samples


def langevin_algorithm_base(x0: torch.Tensor,
                            flow: Flow,
                            target: callable,
                            n_jumps: int = 25,
                            n_trajectories_per_jump: int = 10,
                            batch_size: int = 128,
                            burnin: int = 1000,
                            nf_adjustment: bool = True,
                            show_progress: bool = True,
                            **kwargs):
    # Burnin with standard LMC
    output = base_langevin(
        x0=x0,
        n_iterations=burnin,
        target=target,
        **kwargs
    )

    nf_lmc = NFLMC(
        target_potential=target,
        flow=flow,
        n_jumps=n_jumps,
        jump_period=n_trajectories_per_jump,
        show_progress=show_progress,
        flow_adjustment=nf_adjustment,
        **output.kernel
    )

    xs = nf_lmc.sample(output.samples[-1], flow_fit_kwargs={"batch_size": batch_size})
    return MCMCOutput(samples=xs)


def unadjusted_langevin_algorithm_base(*args, **kwargs):
    return langevin_algorithm_base(*args, **kwargs, adjustment=False)


def metropolis_adjusted_langevin_algorithm_base(*args, **kwargs):
    return langevin_algorithm_base(*args, **kwargs, adjustment=True)
