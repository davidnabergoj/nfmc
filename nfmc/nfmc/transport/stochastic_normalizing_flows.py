import math
from copy import deepcopy
from typing import Sequence

import torch
import torch.nn as nn
import torch.optim as optim

from nfmc.mcmc.mh import mh_step
from normalizing_flows.bijections import RealNVP
from normalizing_flows.bijections.base import Bijection
from potentials.base import Potential
from nfmc.mcmc.hmc import hmc_trajectory


class SNFLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, potential: callable):
        raise NotImplementedError


class MALALayer(SNFLayer):
    def __init__(self,
                 time_step: float = 1.0,
                 friction: float = 1.0,
                 mass: float = 1.0,
                 beta: float = 1.0):
        super().__init__()
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
        delta_s = -0.5 * (eta_tilde.square().sum(dim=-1) - eta.square().sum(dim=-1))
        return x_prime, delta_s


class MCMCLayer(SNFLayer):
    def __init__(self):
        super().__init__()

    def trajectory(self, x, potential: callable):
        raise NotImplementedError

    def forward(self, x, potential: callable):
        x_prime = self.trajectory(x, potential)
        delta_s = potential(x_prime) - potential(x)
        return x, delta_s


class HMCLayer(MCMCLayer):
    def __init__(self, n_leapfrog_steps: int = 10):
        super().__init__()
        self.n_leapfrog_steps = n_leapfrog_steps

    def trajectory(self, x, potential: callable):
        # Keeping the kernel parameters constant to avoid problems.
        # TODO allow user to specify kernel parameters or tune them
        n_dim = x.shape[-1]
        inv_mass_diag = torch.ones(size=(1, n_dim))
        step_size = n_dim ** (-1 / 4)

        momentum = torch.randn_like(x)
        x_prime, _ = hmc_trajectory(
            x=x,
            momentum=momentum,
            inv_mass_diag=inv_mass_diag,
            step_size=step_size,
            n_leapfrog_steps=self.n_leapfrog_steps,
            potential=potential
        )
        return x_prime


class MHLayer(MCMCLayer):
    def __init__(self):
        super().__init__()

    def trajectory(self, x, potential: callable):
        return mh_step(x=x)


class FlowLayer(SNFLayer):
    """
    Thin bijection wrapper that has the additional potential argument in the __call__ signature.
    This argument is unused, but it helps use SNFLayer objects for the entire SNF.
    """

    def __init__(self, bijection: Bijection):
        super().__init__()
        self.bijection = bijection

    def forward(self, x, potential: callable):
        return self.bijection.forward(x)


class SNF(nn.Module):
    def __init__(self, layers: Sequence[SNFLayer], target_potential: Potential, prior_potential: Potential):
        super().__init__()
        assert len(layers) >= 1
        self.layers: nn.ModuleList[SNFLayer] = nn.ModuleList(*layers)
        self.target_potential = target_potential
        self.prior_potential = prior_potential

    def inverse(self, z):
        """
        :param z:
        :return: x and log_weights
        """
        n_steps = len(self.layers)
        lambdas = torch.linspace(1 / n_steps, 1, n_steps)
        # event_shape = z.shape[-1]  # TODO allow the user to specify this
        batch_shape = z.shape[:-1]

        log_det = torch.zeros(size=batch_shape)
        x = deepcopy(z.detach())
        for i, layer in enumerate(self.layers):
            x, delta_s = layer(
                x,
                potential=lambda v: (1 - lambdas[i]) * self.prior_potential(v) + lambdas[i] * self.target_potential(v)
            )
            log_det += delta_s
        log_weights = -self.target_potential(x) + self.prior_potential(z) + log_det
        return x, log_weights

    def fit(self, z, n_epochs: int = 100):
        optimizer = optim.AdamW(self.parameters())

        # Train the SNF
        for _ in range(n_epochs):
            optimizer.zero_grad()

            # Compute loss
            x, log_weights = self.inverse(z)
            loss = torch.mean(-log_weights)

            # Backprop and step
            loss.backward()
            optimizer.step()


def _snf_base(z: torch.Tensor, flow: SNF, **kwargs):
    flow.fit(z, **kwargs)
    with torch.no_grad():
        x, _ = flow.inverse(z)
    return x


def stochastic_normalizing_flow_hmc_base(prior_samples: torch.Tensor,
                                         prior_potential: Potential,
                                         target_potential: Potential,
                                         flow_name: str,
                                         **kwargs):
    n_dim = prior_samples.shape[-1]  # We assume the event is the last dimension

    if flow_name is None:
        return snf_hmc_real_nvp(prior_samples, prior_potential, target_potential, **kwargs)

    # Reasonable default SNF
    if flow_name == "realnvp":
        flow = SNF(
            prior_potential=prior_potential,
            target_potential=target_potential,
            layers=[
                HMCLayer(),
                FlowLayer(RealNVP(n_dim, n_layers=2)),
                HMCLayer(),
                FlowLayer(RealNVP(n_dim, n_layers=2)),
                HMCLayer()
            ]
        )
    else:
        raise ValueError

    return _snf_base(prior_samples, flow, **kwargs)


def snf_hmc_real_nvp(prior_samples: torch.Tensor,
                     prior_potential: Potential,
                     target_potential: Potential,
                     **kwargs):
    n_dim = prior_samples.shape[-1]  # We assume the event is the last dimension

    # Reasonable default SNF
    flow = SNF(
        prior_potential=prior_potential,
        target_potential=target_potential,
        layers=[
            HMCLayer(),
            FlowLayer(RealNVP(n_dim, n_layers=2)),
            HMCLayer(),
            FlowLayer(RealNVP(n_dim, n_layers=2)),
            HMCLayer()
        ]
    )
    return _snf_base(prior_samples, flow, **kwargs)
