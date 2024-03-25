import torch

from nfmc.sampling_implementations.jump.base import JumpMCMC
from normalizing_flows import Flow
from nfmc.mcmc.hmc import hmc


class NFHMC(JumpMCMC):
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
        return "NF-HMC"

    def sample_mcmc(self, x: torch.Tensor):
        x, kernel_params = hmc(
            x0=x,
            n_iterations=self.jump_period - 1,
            potential=self.target_potential,
            full_output=True,
            return_kernel_parameters=True,
            **{
                **self.mcmc_kwargs,
                **{
                    "inv_mass_diag": self.inv_mass_diag,
                    "step_size": self.step_size
                }
            }  # Reuse old kernel parameters and have them overwrite whatever is in kwargs
        )  # (n_steps, n_chains, *event_shape) where n_steps = self.jump_period - 1
        self.inv_mass_diag = kernel_params["inv_mass_diag"]
        self.step_size = kernel_params["step_size"]
        return x


def hmc_base(x0: torch.Tensor,
             flow: Flow,
             potential: callable,
             n_jumps: int = 25,
             n_mcmc_steps_per_jump: int = 5,  # 5 HMC trajectories between jumps
             batch_size: int = 128,
             burnin: int = 1000,
             adjustment: bool = False,  # HMC adjustment
             nf_adjustment: bool = True,  # Flow jump adjustment
             show_progress: bool = True,
             **kwargs):
    # Burnin with standard HMC
    x, kernel_params = hmc(
        x0=x0,
        n_iterations=burnin,
        full_output=False,
        potential=potential,
        return_kernel_parameters=True,
        adjustment=adjustment,
        show_progress=True,
        **kwargs
    )

    nf_hmc = NFHMC(
        target_potential=potential,
        flow=flow,
        n_jumps=n_jumps,
        jump_period=n_mcmc_steps_per_jump,
        show_progress=show_progress,
        flow_adjustment=nf_adjustment,
        inv_mass_diag=kernel_params["inv_mass_diag"],
        step_size=kernel_params["step_size"]
    )

    x = nf_hmc.sample(x, flow_fit_kwargs={"batch_size": batch_size})
    return x


def unadjusted_hmc_base(*args, **kwargs):
    return hmc_base(*args, **kwargs, adjustment=False)


def adjusted_hmc_base(*args, **kwargs):
    return hmc_base(*args, **kwargs, adjustment=True)
