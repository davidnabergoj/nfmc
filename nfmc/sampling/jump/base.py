from copy import deepcopy

import torch

from nfmc.util import metropolis_acceptance_log_ratio
from normalizing_flows import Flow


class JumpMCMC:
    """
    Base class for MCMC interleaved with NF-based jumps.
    """

    def __init__(self,
                 target_potential: callable,
                 flow: Flow,
                 n_jumps: int = 100,
                 jump_period: int = 500,
                 flow_adjustment: bool = True,
                 show_progress: bool = True):
        """
        :param target_potential: function that receives as input a batch of states with shape (batch_size, *event_shape)
            and returns the potential of each individual state in a tensor with shape (batch_size,).
        :param flow: normalizing flow object with the following methods:
            - Flow.fit(states),
            - Flow.sample(n_new_states),
            - Flow.log_prob(states)
        """
        self.target_potential = target_potential
        self.flow = flow

        self.n_jumps = n_jumps
        self.jump_period = jump_period
        self.flow_adjustment = flow_adjustment
        self.show_progress = show_progress

    @property
    def name(self) -> str:
        raise NotImplementedError

    @staticmethod
    def update_progress_bar(progress_bar,
                            n_current_accepted: int,
                            n_chains: int,
                            total_accepted: int,
                            total_seen: int):
        progress_bar.set_postfix_str(
            f'current accepted fraction: {n_current_accepted / n_chains:.3f}, '
            f'total accepted fraction: {total_accepted / total_seen:.3f}, '
        )

    def sample_mcmc(self, x: torch.Tensor):
        raise NotImplementedError

    def sample(self, x0: torch.Tensor, flow_fit_kwargs: dict = None):
        """
        It is assumed that the states x0 have already burned in.
        """
        if flow_fit_kwargs is None:
            flow_fit_kwargs = dict()

        n_chains, *event_shape = x0.shape
        x = deepcopy(x0)
        if not torch.all(torch.isfinite(x)):
            raise ValueError("All initial states must be finite")

        # Fit the flow to the burned in states
        self.flow.fit(x, **flow_fit_kwargs)
        # We want a decent fit at this point.

        if self.show_progress:
            from tqdm import tqdm
            progress_bar = tqdm(
                range(self.n_jumps),
                desc=f'{self.name} '
                     f'({n_chains} chains, '
                     f'{self.jump_period} MCMC iterations per jump, '
                     f'{"adjusted" if self.flow_adjustment else "unadjusted"} jumps)'
            )
        else:
            progress_bar = range(self.n_jumps)

        total_accepted = 0
        total_seen = 0
        xs = []
        for _ in progress_bar:
            x_mcmc = self.sample_mcmc(x)
            if not torch.all(torch.isfinite(x_mcmc)):
                raise ValueError("Sampling diverged, found state with nan/inf elements")
            xs.append(x_mcmc)

            x_train = x_mcmc.view(-1, *event_shape)  # (n_steps * n_chains, *event_shape)
            self.flow.fit(x_train, n_epochs=1, shuffle=False, **flow_fit_kwargs)
            x_proposed = self.flow.sample(n_chains).detach().cpu()  # (n_chains, *event_shape)
            if self.flow_adjustment:
                x_current = x_mcmc[-1]
                u_x = self.target_potential(x_current)
                u_x_prime = self.target_potential(x_proposed)
                f_x = self.flow.log_prob(x_current)
                f_x_prime = self.flow.log_prob(x_proposed)
                log_alpha = metropolis_acceptance_log_ratio(
                    log_prob_curr=-u_x,
                    log_prob_prime=-u_x_prime,
                    log_proposal_curr=f_x,
                    log_proposal_prime=f_x_prime
                ).cpu()
                acceptance_mask = torch.rand_like(log_alpha).log() < log_alpha
                x[acceptance_mask] = x_proposed[acceptance_mask]
                n_current_accepted = int(torch.sum(torch.as_tensor(acceptance_mask).float()))
                total_accepted += n_current_accepted
            else:
                x = x_proposed
                n_current_accepted = n_chains
                total_accepted += n_current_accepted
            total_seen += n_chains
            if self.show_progress:
                self.update_progress_bar(
                    progress_bar,
                    n_current_accepted,
                    n_chains,
                    total_accepted,
                    total_seen
                )
            # x.shape = (n_chains, n_dim)
            xs.append(deepcopy(x)[None])
        return torch.cat(xs, dim=0)
