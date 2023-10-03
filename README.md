# Normalizing flow Monte Carlo

This package provides various Markov chain Monte Carlo (MCMC) algorithms that utilize normalizing flows (NF).
These are listed in the following table.
The last column specifies whether the algorithm relies on both forward and inverse maps of the underlying NF bijection
or only one.

| Sampler                                                | Abbreviation |         Kind          | Requires NF forward and inverse | Requires prior | 
|--------------------------------------------------------|:------------:|:---------------------:|:-------------------------------:|:--------------:|
| Independent Metropolis Hastings                        |     IMH      |       Sampling        |               Yes               |       No       |
| Metropolis adjusted Langevin algorithm                 |   NF-MALA    |       Sampling        |               Yes               |       No       |
| Unadjusted Langevin algorithm                          |    NF-ULA    |       Sampling        |               Yes               |       No       |
| NeuTra HMC                                             |    NeuTra    |       Sampling        |               No                |       No       |
| Transport elliptical slice sampling                    |     TESS     |       Sampling        |               Yes               |       No       |
| Nested sampling                                        |    NESSAI    |       Transport       |               Yes               |      Yes       |
| Deterministic Langevin Monte Carlo                     |     DLMC     |       Transport       |               *No               |      Yes       |
| Stochastic normalizing flows                           |     SNF      |       Transport       |               No                |      Yes       |
| Preconditioned Monte Carlo                             |     PMC      |       Transport       |               Yes               |      Yes       |
| Annealed flow transport Monte Carlo                    |     AFT      |       Transport       |               No                |      Yes       |
| Continual repeated annealed flow transport Monte Carlo |    CRAFT     |       Transport       |               No                |      Yes       |
| Flow annealed importance sampling bootstrap            |     FAB      | Variational inference |               Yes               |       No       |

&ast; Standard DLMC only requires the forward map, but latent DLMC requires the inverse map as well. 



## Example use

Sampling NFMC algorithms require a target potential and a normalizing flow object.
They output draws with shape `(n_iterations, n_chains, *event_shape)`.
An example using the Real NVP flow and a standard Gaussian potential is shown below:
```python
import torch
from potentials.synthetic.gaussian.unit import StandardGaussian
from nfmc.sampling import mala, ula, imh, neutra_hmc, tess

torch.manual_seed(0)  # Set the random seed for reproducible results

n_iterations = 1000
n_chains = 100
event_shape = (25,)  # Each draw (event) is a vector with size 25.

# Define the target potential
target = StandardGaussian(event_shape=event_shape)

# Draw samples with different NFMC sampling methods
mala_samples = mala(target, "realnvp", n_chains, n_iterations)
ula_samples = ula(target, "realnvp", n_chains, n_iterations)
imh_samples = imh(target, "realnvp", n_chains, n_iterations)
neutra_samples = neutra_hmc(target, "realnvp", n_chains, n_iterations)
tess_samples = tess(target, "realnvp", n_chains, n_iterations)

print(mala_samples.shape)  # (1000, 100, 25)
print(ula_samples.shape)  # (1000, 100, 25)
print(imh_samples.shape)  # (1000, 100, 25)
print(neutra_samples.shape)  # (1000, 100, 25)
print(tess_samples.shape)  # (1000, 100, 25)
```

Transport NFMC algorithms move particles from a prior potential to a target potential.
They output a particle history with shape `(n_iterations, n_particles, *event_shape)`.
The last iteration represents particles that are distributed according to the target.
An example using the Real NVP flow, a standard Gaussian prior potential, and a diagonal Gaussian target potential is shown below:
```python
import torch
from potentials.synthetic.gaussian.unit import StandardGaussian
from potentials.synthetic.gaussian.diagonal import DiagonalGaussian0
from nfmc.transport import aft, craft, snf, ns, dlmc

torch.manual_seed(0)  # Set the random seed for reproducible results

n_iterations = 1000
n_particles = 100
event_shape = (25,)  # Each draw (event) is a vector with size 25.

# Define the target potential
prior = StandardGaussian(event_shape)
target = DiagonalGaussian0(event_shape)

# Draw samples with different NFMC sampling methods
snf_samples = snf(prior, target, "realnvp", n_particles, n_iterations)
aft_samples = aft(prior, target, "realnvp", n_particles, n_iterations)
craft_samples = craft(prior, target, "realnvp", n_particles, n_iterations)
dlmc_samples = dlmc(prior, target, "realnvp", n_particles, n_iterations)
ns_samples = ns(prior, target, "realnvp", n_particles, n_iterations)

print(snf_samples.shape)  # (1000, 100, 25)
print(aft_samples.shape)  # (1000, 100, 25)
print(craft_samples.shape)  # (1000, 100, 25)
print(dlmc_samples.shape)  # (1000, 100, 25)
print(ns_samples.shape)  # (1000, 100, 25)
```