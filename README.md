# Markov Chain Monte Carlo with Normalizing Flows

This package provides various sampling algorithms that utilize normalizing flows (NF):

* Independent Metropolis Hastings (IMH)
* Metropolis adjusted Langevin algorithm (MALA)
* Unadjusted Langevin algorithm (MALA)
* NeuTra HMC
* Transport elliptical slice sampling (TESS)
* Deterministic Langevin Monte Carlo (DLMC)

The following algorithms are yet to be added:

* Preconditioned Monte Carlo (PMC)
* Flow annealed importance sampling bootstrap (FAB)
* Nested sampling (NS)
* Stochastic normalizing flows (SNF)
* Annealed flow transport Monte Carlo (AFT)
* Continual repeated annealed flow transport Monte Carlo (CRAFT)

We term such NF-based MCMC algorithms as NFMC.

---

## Usage instructions

NFMC algorithms require a target potential and an NF object.
The potential is a function that computes the negative unnormalized log probability density of the target distribution.
Example potentials are provided in the accompanying [potentials package](https://github.com/davidnabergoj/potentials).
This package depends on [torchflows](https://github.com/davidnabergoj/torchflows) for NF definitions.
Please implement custom NF architectures in torchflows for compatibility with this package.

An example using Real NVP and a standard Gaussian potential is shown below.

```python
import torch
from nfmc import sample

torch.manual_seed(0)  # Set the random seed for reproducible results

n_iterations = 1000
n_chains = 100
n_dim = 25  # Each draw (event) is a vector with size 25.


# Define the target potential
def standard_gaussian_potential(x):
    return torch.sum(x ** 2, dim=-1)


# Draw samples with Jump MALA (also local-global MALA)
mala_out = sample(
    standard_gaussian_potential,
    event_shape=(n_dim,),
    strategy="jump_mala",
    flow="realnvp",
    n_chains=n_chains,
    n_iterations=n_iterations
)
```

To use different samplers, pass the following keyword arguments to the `sample` function:

```python
from nfmc import sample

sample(..., strategy='jump_mh')         # Metropolis-Hastings with NF jumps
sample(..., strategy='imh')             # Independent Metropolis-Hastings
sample(..., strategy='adaptive_imh')    # Adaptive independent Metropolis-Hastings
sample(..., strategy='jump_mala')       # Metropolis adjusted Langevin algorithm with NF jumps
sample(..., strategy='jump_ula')        # Unadjusted Langevin algorithm with NF jumps
sample(..., strategy='jump_hmc')        # Hamiltonian Monte Carlo with NF jumps
sample(..., strategy='jump_uhmc')       # Unadjusted Hamiltonian Monte Carlo with NF jumps
sample(..., strategy='jump_ess')        # Elliptical slice sampling with NF jumps
sample(..., strategy='jump_ess')        # Elliptical slice sampling with NF jumps
sample(..., strategy='tess')            # Transport elliptical slice sampling
sample(..., strategy='dlmc')            # Deterministic Langevin Monte Carlo
sample(..., strategy='neutra_hmc')      # NeuTra HMC
sample(..., strategy='neutra_mh')       # NeuTra MH
```

The output of the `sample` method is an `MCMCOutput` object with useful properties for model analyses and sampler debugging:

```python
out = sample(...)

out.samples         # tensor of samples
out.mean            # mean tensor
out.variance        # variance tensor
out.second_moment   # second moment tensor
out.statistics      # MCMCStatistics object with convergence monitoring logs and other MCMC-related quantities (e.g., acceptance rate, number of target calls, number of target gradient calls)  
``` 

Other samplers may be used by explicitly creating the sampler object and calling the `sample` method.
We provide an example for Jump HMC and the standard Gaussian potential:

```python
import torch
from nfmc.algorithms.sampling.nfmc.jump import JumpHMC

torch.manual_seed(0)
event_shape = (10,)


def standard_gaussian_potential(x):
    return torch.sum(x ** 2, dim=-1)


sampler = JumpHMC(event_shape, standard_gaussian_potential)

n_chains = 100
x_initial = torch.randn(size=(n_chains, *event_shape)) / 10
out = sampler.sample(x_initial)
```

You can check the list of supported NFs with:

```python
from nfmc.util import get_supported_normalizing_flows

print(get_supported_normalizing_flows())
```

---

## Setup

This package was tested with Python version 3.10, however we expect Python versions 3.7+ to also work.
This package depends on [torchflows](https://github.com/davidnabergoj/torchflows).

Clone the package and install dependencies:

```
git clone git@github.com:davidnabergoj/nfmc.git
pip install torchflows
```

---

## Contributing

We warmly welcome any contributions or comments.
Some aspects of the package that can be improved:

* Additional sampler tests.
* Implementation of PMC, NS, SNF, AFT, CRAFT, FAB and other NFMC methods.
* Configuring a continuous integration pipeline with Github actions.
* Suggestions for improved default sampler hyperparameters.

## Citation

If you use this code in your work, we kindly ask that you cite the accompanying paper:

```

```