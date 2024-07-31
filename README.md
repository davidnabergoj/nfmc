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
| Deterministic Langevin Monte Carlo                     |     DLMC     |       Sampling        |               *No               |      Yes       |
| Nested sampling                                        |    NESSAI    |       Transport       |               Yes               |      Yes       |
| Stochastic normalizing flows                           |     SNF      |       Transport       |               No                |      Yes       |
| Preconditioned Monte Carlo                             |     PMC      |       Transport       |               Yes               |      Yes       |
| Annealed flow transport Monte Carlo                    |     AFT      |       Transport       |               No                |      Yes       |
| Continual repeated annealed flow transport Monte Carlo |    CRAFT     |       Transport       |               No                |      Yes       |
| Flow annealed importance sampling bootstrap            |     FAB      | Variational inference |               Yes               |       No       |

&ast; Standard DLMC only requires the forward map, but latent DLMC requires the inverse map as well.

## Example use

Sampling NFMC algorithms require a target potential and a normalizing flow object.
IMH, NeuTra and TESS output draws with shape `(n_iterations, n_chains, *event_shape)`.
MALA and ULA output draws with shape `(n_iterations * jump_period, n_chains, *event_shape)`.
An example using the Real NVP flow and a standard Gaussian potential is shown below.

```python
import torch
from potentials.synthetic.gaussian.unit import StandardGaussian
from nfmc import sample

torch.manual_seed(0)  # Set the random seed for reproducible results

n_iterations = 1000
n_chains = 100
n_dim = 25  # Each draw (event) is a vector with size 25.

# Define the target potential
target = StandardGaussian(n_dim=n_dim)

# Draw samples with different NFMC sampling methods
mala_out = sample(
    target,
    strategy="jump_mala",
    flow="realnvp",
    n_chains=n_chains,
    n_iterations=n_iterations
)
ula_out = sample(
    target,
    strategy="jump_ula",
    flow="realnvp",
    n_chains=n_chains,
    n_iterations=n_iterations
)
imh_out = sample(
    target,
    strategy="imh",
    flow="realnvp",
    n_chains=n_chains,
    n_iterations=n_iterations
)
neutra_out = sample(
    target,
    strategy="neutra_hmc",
    flow="realnvp",
    n_chains=n_chains,
    n_iterations=n_iterations
)
tess_out = sample(
    target,
    strategy="tess",
    flow="realnvp",
    n_chains=n_chains,
    n_iterations=n_iterations
)
```

You can check the supported normalizing flows with:

```python
from nfmc.util import get_supported_normalizing_flows

print(get_supported_normalizing_flows())
```