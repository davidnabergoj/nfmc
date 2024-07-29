import math
from copy import deepcopy
from typing import Sequence
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from nfmc.mcmc.mh import mh
from normalizing_flows.bijections import RealNVP
from normalizing_flows.bijections.base import Bijection
from potentials.base import Potential
from nfmc.mcmc.hmc import hmc
from normalizing_flows.utils import sum_except_batch


class SNFLayer(nn.Module):
    def __init__(self, event_shape):
        super().__init__()
        self.event_shape = event_shape

    def forward(self, x: torch.Tensor, potential: callable):
        raise NotImplementedError


class MALALayer(SNFLayer):
    def __init__(self,
                 event_shape,
                 time_step: float = 1.0,
                 friction: float = 1.0,
                 mass: float = 1.0,
                 beta: float = 1.0):
        super().__init__(event_shape)
        self.beta = beta
        self.eps = time_step / (friction * mass)

    def forward(self, x, potential: callable):
        with torch.enable_grad():
            grad_x = torch.autograd.grad(potential(x).sum(), x)[0].detach()
        assert grad_x.shape == x.shape
        eta = torch.randn_like(x)
        x_prime = x - self.eps * grad_x + math.sqrt(2 * self.eps / self.beta) * eta
        with torch.enable_grad():
            grad_x_prime = torch.autograd.grad(potential(x_prime).sum(), x_prime)[0].detach()
        eta_tilde = math.sqrt(2 * self.eps / self.beta) * (grad_x + grad_x_prime) - eta
        # TODO sum over event dims i.e. sum_except_batch
        # delta_s = -0.5 * (eta_tilde.square().sum(dim=-1) - eta.square().sum(dim=-1))
        delta_s = -0.5 * (
                sum_except_batch(eta_tilde.square(), self.event_shape)
                - sum_except_batch(eta.square(), self.event_shape)
        )
        return x_prime, delta_s


class MCMCLayer(SNFLayer):
    def __init__(self, event_shape):
        super().__init__(event_shape)

    def sample(self, x, potential: callable, **kwargs):
        raise NotImplementedError

    def forward(self, x, potential: callable):
        x_prime = self.sample(x, potential)
        energy_delta = potential(x_prime) - potential(x)
        return x_prime, energy_delta


class HMCLayer(MCMCLayer):
    def __init__(self, event_shape):
        super().__init__(event_shape)

    def sample(self, x, potential: callable, **kwargs):
        x = hmc(
            x0=x,
            target=potential,
            full_output=False,
            n_iterations=100,
            **kwargs
        )
        return x


class MHLayer(MCMCLayer):
    def __init__(self, event_shape):
        super().__init__(event_shape)

    def sample(self, x, potential: callable, **kwargs):
        return mh(x, potential, **kwargs)


class FlowLayer(SNFLayer):
    """
    Thin bijection wrapper that has the additional potential argument in the __call__ signature.
    This argument is unused, but it helps use SNFLayer objects for the entire SNF.
    """

    def __init__(self, bijection: Bijection):
        super().__init__(bijection.event_shape)
        self.bijection = bijection

    def forward(self, x, potential: callable):
        return self.bijection.forward(x)


class SNF(nn.Module):
    def __init__(self, layers: Sequence[SNFLayer], target_potential: Potential, prior_potential: Potential):
        super().__init__()
        assert len(layers) >= 1
        self.layers: nn.ModuleList[SNFLayer] = nn.ModuleList(layers)
        self.target_potential = target_potential
        self.prior_potential = prior_potential

    def inverse(self, z):
        """
        :param z:
        :return: x and log_weights
        """
        n_steps = len(self.layers)
        lambdas = torch.linspace(1 / n_steps, 1, n_steps)
        batch_shape = (z.shape[0],)  # Assuming this

        log_det = torch.zeros(size=batch_shape)
        x = deepcopy(z.detach())

        particle_history = [x]
        for i, layer in enumerate(self.layers):
            intermediate_potential = lambda v: (
                    (1 - lambdas[i]) * self.prior_potential(v)
                    + lambdas[i] * self.target_potential(v)
            )

            x, delta_s = layer(x, potential=intermediate_potential)
            log_det += delta_s
            particle_history.append(x)
        log_weights = -self.target_potential(x) + self.prior_potential(z) + log_det
        particle_history = torch.stack(particle_history)
        return particle_history, x, log_weights

    def fit(self, z, n_epochs: int = 10, show_progress: bool = True):
        optimizer = optim.AdamW(self.parameters())

        if show_progress:
            iterator = tqdm(range(n_epochs), desc="SNF")
        else:
            iterator = range(n_epochs)

        # Train the SNF
        for _ in iterator:
            optimizer.zero_grad()

            # Compute loss
            _, x, log_weights = self.inverse(z)
            loss = torch.mean(-log_weights)

            # Backprop and step
            loss.backward()
            optimizer.step()


def _snf_base(z: torch.Tensor, flow: SNF, **kwargs):
    flow.fit(z, **kwargs)
    with torch.no_grad():
        particle_history, _, _ = flow.inverse(z)
    return particle_history


def stochastic_normalizing_flow_hmc_base(prior_samples: torch.Tensor,
                                         prior_potential: Potential,
                                         target_potential: Potential,
                                         flow_name: str,
                                         **kwargs):
    if flow_name is None:
        return snf_hmc_real_nvp(prior_samples, prior_potential, target_potential, **kwargs)

    event_shape = prior_potential.event_shape

    # Reasonable default SNF
    if flow_name == "realnvp":
        flow = SNF(
            prior_potential=prior_potential,
            target_potential=target_potential,
            layers=[
                HMCLayer(event_shape),
                FlowLayer(RealNVP(event_shape, n_layers=2)),
                HMCLayer(event_shape),
                FlowLayer(RealNVP(event_shape, n_layers=2)),
                HMCLayer(event_shape)
            ]
        )
    else:
        raise ValueError

    return _snf_base(prior_samples, flow, **kwargs)


def snf_hmc_real_nvp(prior_samples: torch.Tensor,
                     prior_potential: Potential,
                     target_potential: Potential,
                     **kwargs):
    event_shape = prior_potential.event_shape

    # Reasonable default SNF
    flow = SNF(
        prior_potential=prior_potential,
        target_potential=target_potential,
        layers=[
            HMCLayer(event_shape),
            FlowLayer(RealNVP(event_shape, n_layers=2)),
            HMCLayer(event_shape),
            FlowLayer(RealNVP(event_shape, n_layers=2)),
            HMCLayer(event_shape)
        ]
    )
    return _snf_base(prior_samples, flow, **kwargs)
