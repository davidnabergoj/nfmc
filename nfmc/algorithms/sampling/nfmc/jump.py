from dataclasses import dataclass
from typing import Sized

import torch

from nfmc.algorithms.sampling.base import Sampler, MCMCOutput, NFMCParameters, NFMCKernel, MCMCStatistics, MCMCKernel, \
    MCMCParameters
from tqdm import tqdm

from nfmc.algorithms.sampling.mcmc import HMC, UHMC, MALA, ULA, MH, NUTS
from nfmc.algorithms.sampling.mcmc.ess import ESS
from nfmc.algorithms.sampling.tuning import train_val_split
from nfmc.util import metropolis_acceptance_log_ratio


@dataclass
class JumpNFMCParameters(NFMCParameters):
    adjusted_jumps: bool = True
    fit_nf: bool = False


@dataclass
class JumpNFMCStatistics(MCMCStatistics):
    n_accepted_jumps: int = 0
    n_attempted_jumps: int = 0
    n_divergences: int = 0

    @property
    def jump_acceptance_rate(self):
        if self.n_attempted_jumps == 0:
            return torch.nan
        return self.n_accepted_jumps / self.n_attempted_jumps

    def __repr__(self):
        return (f'mcmc acceptance rate: {self.acceptance_rate:.3f} | '
                f'jump acceptance rate: {self.jump_acceptance_rate:.3f}')


class JumpNFMC(Sampler):
    def __init__(self,
                 event_shape: Sized,
                 target: callable,
                 inner_sampler: Sampler,
                 kernel: NFMCKernel = None,
                 params: JumpNFMCParameters = None):
        if kernel is None:
            kernel = NFMCKernel(event_shape)
        if params is None:
            params = JumpNFMCParameters()
        super().__init__(event_shape, target, kernel, params)
        self.inner_sampler = inner_sampler

    def warmup(self, x0: torch.Tensor, show_progress: bool = True) -> MCMCOutput:
        self.params: JumpNFMCParameters
        mcmc_output = self.inner_sampler.warmup(x0, show_progress=True)
        x_train, x_val = train_val_split(
            mcmc_output.samples,
            train_pct=self.params.train_pct,
            max_train_size=self.params.max_train_size,
            max_val_size=self.params.max_val_size
        )
        self.kernel.flow.fit(x_train=x_train, x_val=x_val, **self.params.flow_fit_kwargs)

    def sample(self, x0: torch.Tensor, show_progress: bool = True) -> MCMCOutput:
        self.kernel: NFMCKernel
        self.params: JumpNFMCParameters

        n_chains, *event_shape = x0.shape
        xs = torch.zeros(
            size=(self.params.n_iterations, self.inner_sampler.params.n_iterations + 1, *x0.shape),
            device=x0.device,
            dtype=x0.dtype
        )  # (jumps, trajectories per jump, chains, *event)
        statistics = JumpNFMCStatistics()

        x = torch.clone(x0)
        for i in (pbar := tqdm(range(self.params.n_iterations), desc='Jump MCMC', disable=not show_progress)):
            # Trajectories
            pbar.set_description_str(f'[Jump MCMC] sampling')
            mcmc_output = self.inner_sampler.sample(x0=x, show_progress=False)
            statistics.n_accepted_trajectories += mcmc_output.statistics.n_accepted_trajectories
            statistics.n_attempted_trajectories += mcmc_output.statistics.n_attempted_trajectories
            statistics.n_divergences += mcmc_output.statistics.n_divergences

            xs[i, :-1] = mcmc_output.samples

            # Fit flow
            if self.params.fit_nf:
                pbar.set_description_str(f'[Jump MCMC] training')
                x_train, x_val = train_val_split(
                    xs,
                    train_pct=self.params.train_pct,
                    max_train_size=self.params.max_train_size,
                    max_val_size=self.params.max_val_size
                )
                self.kernel.flow.fit(x_train=x_train, x_val=x_val, **self.params.flow_fit_kwargs)

            # Jump
            pbar.set_description_str(f'[Jump MCMC] jumping')
            x_prime = self.kernel.flow.sample(n_chains).detach()
            x = mcmc_output.samples[-1]
            if self.params.adjusted_jumps:
                try:
                    u_x = self.target(x)
                    u_x_prime = self.target(x_prime)
                    f_x = self.kernel.flow.log_prob(x)
                    f_x_prime = self.kernel.flow.log_prob(x_prime)
                    log_alpha = metropolis_acceptance_log_ratio(
                        log_prob_curr=-u_x,
                        log_prob_prime=-u_x_prime,
                        log_proposal_curr=f_x,
                        log_proposal_prime=f_x_prime
                    )
                    acceptance_mask = torch.rand_like(log_alpha).log() < log_alpha
                except ValueError:
                    acceptance_mask = torch.zeros(size=x.shape[:-len(event_shape)], dtype=torch.bool)
            else:
                acceptance_mask = torch.ones(size=x.shape[:-len(event_shape)], dtype=torch.bool)
            x[acceptance_mask] = x_prime[acceptance_mask]
            statistics.n_attempted_jumps += n_chains
            statistics.n_accepted_jumps += int(torch.sum(acceptance_mask))
            pbar.set_postfix_str(f'{statistics}')
            xs[i, -1] = x
        xs = xs.flatten(0, 1)
        return MCMCOutput(samples=xs, statistics=statistics)


class JumpHMC(JumpNFMC):
    def __init__(self,
                 event_shape: Sized,
                 target: callable,
                 kernel: NFMCKernel = None,
                 params: JumpNFMCParameters = None,
                 inner_kernel: MCMCKernel = None,
                 inner_params: MCMCParameters = None):
        inner_sampler = HMC(event_shape, target, inner_kernel, inner_params)
        super().__init__(event_shape, target, inner_sampler, kernel, params)


class JumpUHMC(JumpNFMC):
    def __init__(self,
                 event_shape: Sized,
                 target: callable,
                 kernel: NFMCKernel = None,
                 params: JumpNFMCParameters = None,
                 inner_kernel: MCMCKernel = None,
                 inner_params: MCMCParameters = None):
        inner_sampler = UHMC(event_shape, target, inner_kernel, inner_params)
        super().__init__(event_shape, target, inner_sampler, kernel, params)


class JumpMALA(JumpNFMC):
    def __init__(self,
                 event_shape: Sized,
                 target: callable,
                 kernel: NFMCKernel = None,
                 params: JumpNFMCParameters = None,
                 inner_kernel: MCMCKernel = None,
                 inner_params: MCMCParameters = None):
        inner_sampler = MALA(event_shape, target, inner_kernel, inner_params)
        super().__init__(event_shape, target, inner_sampler, kernel, params)


class JumpULA(JumpNFMC):
    def __init__(self,
                 event_shape: Sized,
                 target: callable,
                 kernel: NFMCKernel = None,
                 params: JumpNFMCParameters = None,
                 inner_kernel: MCMCKernel = None,
                 inner_params: MCMCParameters = None):
        inner_sampler = ULA(event_shape, target, inner_kernel, inner_params)
        super().__init__(event_shape, target, inner_sampler, kernel, params)


class JumpMH(JumpNFMC):
    def __init__(self,
                 event_shape: Sized,
                 target: callable,
                 kernel: NFMCKernel = None,
                 params: JumpNFMCParameters = None,
                 inner_kernel: MCMCKernel = None,
                 inner_params: MCMCParameters = None):
        inner_sampler = MH(event_shape, target, inner_kernel, inner_params)
        super().__init__(event_shape, target, inner_sampler, kernel, params)


class JumpESS(JumpNFMC):
    def __init__(self,
                 event_shape: Sized,
                 target: callable,
                 negative_log_likelihood: callable,
                 kernel: NFMCKernel = None,
                 params: JumpNFMCParameters = None,
                 inner_kernel: MCMCKernel = None,
                 inner_params: MCMCParameters = None):
        inner_sampler = ESS(event_shape, target, negative_log_likelihood, inner_kernel, inner_params)
        super().__init__(event_shape, target, inner_sampler, kernel, params)


class JumpNUTS(JumpNFMC):
    def __init__(self,
                 event_shape: Sized,
                 target: callable,
                 kernel: NFMCKernel = None,
                 params: JumpNFMCParameters = None,
                 inner_kernel: MCMCKernel = None,
                 inner_params: MCMCParameters = None):
        inner_sampler = NUTS(event_shape, target, inner_kernel, inner_params)
        super().__init__(event_shape, target, inner_sampler, kernel, params)
