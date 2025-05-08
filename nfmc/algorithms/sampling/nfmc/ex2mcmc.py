from dataclasses import dataclass
from typing import Tuple, Union
import torch

from nfmc.algorithms.sampling.base import NFMCKernel, Sampler
from nfmc.algorithms.sampling.nfmc.jump import JumpNFMC, JumpNFMCParameters
from nfmc.util import create_flow_object
from nfmc.algorithms.sampling.mcmc.mh import MH, MHKernel, MHParameters
from nfmc.algorithms.sampling.mcmc.hmc import HMC, HMCKernel, HMCParameters
from nfmc.algorithms.sampling.base import MCMCOutput


@dataclass
class Ex2MCMCParameters(JumpNFMCParameters):
    pool_size: int = 20


@dataclass
class Ex2MCMCKernel(NFMCKernel):
    pass


class Ex2MCMC(JumpNFMC):
    def __init__(self,
                 event_shape,
                 target,
                 inner_sampler: Sampler,
                 kernel: Ex2MCMCKernel = None,
                 params: Ex2MCMCParameters = None):
        if kernel is None:
            kernel = Ex2MCMCKernel(event_shape, target,
                                   create_flow_object('realnvp', event_shape))
        if params is None:
            params = Ex2MCMCParameters()
        super().__init__(event_shape, target, inner_sampler, kernel, params)

    def nf_step(self, x, out):
        """
        x.shape = (n_chains, *event_shape)
        should return tensor with shape (n_chains, *event_shape)
        """
        n_chains, *event_shape = x.shape
        flow = self.kernel.flow
        pool_size = self.params.pool_size

        # Prepare tensors for candidates and log prob under the NF
        # candidates.shape = (n_chains, pool_size, event_shape)
        candidates = torch.zeros(
            size=(n_chains, self.params.pool_size, *event_shape)).to(x)
        log_prob_flow = torch.zeros(
            size=(n_chains, self.params.pool_size)).to(x)

        # sample candidates, compute log density under the target and the NF
        candidates[:, 0], log_prob_flow[:, 0] = x, flow.log_prob(x)
        candidates[:, 1:], log_prob_flow[:, 1:] = flow.sample(
            (n_chains, pool_size - 1), return_log_prob=True)
        candidates = candidates.detach()
        log_prob_flow = log_prob_flow.detach()
        log_prob_target = -self.target(candidates)

        out.statistics.update_counters(
            n_target_calls=pool_size * n_chains,
            n_attempted_jumps=n_chains,
            n_accepted_jumps=n_chains,
        )

        # compute self-normalized weights and sample candidates
        log_sn_weights = log_prob_target - log_prob_flow
        pool_indices = torch.tensor([
            int(torch.distributions.Categorical(
                logits=log_sn_weights[i]).sample(()))
            for i in range(len(log_sn_weights))
        ])

        # return chosen points
        return candidates[range(n_chains), pool_indices]


class Ex2MH(Ex2MCMC):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 target: callable,
                 kernel: Ex2MCMCKernel = None,
                 params: Ex2MCMCParameters = None,
                 inner_kernel: MHKernel = None,
                 inner_params: MHParameters = None):
        inner_sampler = MH(event_shape, target, inner_kernel, inner_params)
        super().__init__(event_shape, target, inner_sampler, kernel, params)


class Ex2HMC(Ex2MCMC):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 target: callable,
                 kernel: Ex2MCMCKernel = None,
                 params: Ex2MCMCParameters = None,
                 inner_kernel: HMCKernel = None,
                 inner_params: HMCParameters = None):
        inner_sampler = HMC(event_shape, target, inner_kernel, inner_params)
        super().__init__(event_shape, target, inner_sampler, kernel, params)


class IteratedSIR(Ex2MCMC):
    def __init__(self,
                 event_shape,
                 target,
                 kernel=None,
                 params=None):
        inner_sampler = MH(
            event_shape,
            target,
            params=MHParameters(
                n_iterations=0,
                n_warmup_iterations=0,
                tuning=False,
                store_samples=True,  
                # hack because Jump NFMC requires sample storing. But there are zero iterations so it does not matter.
            )
        )
        super().__init__(event_shape, target, inner_sampler, kernel, params)

    def warmup(self,
               x0: torch.Tensor,
               show_progress: bool = True,
               time_limit_seconds: Union[float, int] = None):
        self.kernel: Ex2MCMCKernel
        self.params: Ex2MCMCParameters

        self.kernel.flow.variational_fit(
            lambda v: -self.target(v),
            **self.params.warmup_fit_kwargs,
            show_progress=show_progress,
            time_limit_seconds=time_limit_seconds
        )
        out = MCMCOutput(
            event_shape=x0.shape[1:], store_samples=self.params.store_samples)
        out.running_samples.add(self.kernel.flow.sample(x0.shape[0]).detach())
        return out
