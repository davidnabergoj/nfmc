# Markov Chain Monte Carlo with Normalizing Flows

This package implements MCMC algorithms that utilize normalizing flows (NF).
Currently, it supports Metropolis-Hastings kernels:

* Independent Metropolis-Hastings (IMH)
* Random walk Metropolis-Hastings (RWMH)
* Metropolis adjusted Langevin algorithm (MALA)
* Hamiltonian Monte Carlo (HMC)

NFs can be used as preconditioners, as in the NeuTra MCMC framework (Hoffman et al., 2017).
They can also be used as independent proposal distributions, as in IMH and Ex2 MCMC (Samsonov et al., 2022).
Alternative options are the diagonal linear and non-diagonal (dense) linear preconditioners or independent proposal distributions.
The samplers are implemented in PyTorch.

This package is intended for research purposes or sampling when the time cost of computing the target probability density is much greater than the time cost of NF operations.

## Installation

This package was tested with Python version 3.10, however we expect Python versions 3.7+ to also work.
This package depends on [torchflows](https://github.com/davidnabergoj/torchflows).

Install the package directly from Github:
```
pip install git+https://github.com/davidnabergoj/nfmc.git
```

To alternatively configure the package for local development, clone the repository and install dependencies:

```
git clone git@github.com:davidnabergoj/nfmc.git
pip install torchflows
```

## Sampling example and usage instructions

Each distribution is defined with a function that computes its (unnormalized) negative log probability density. 
The function accepts an event tensor with shape `(n_chains, *event_shape)` and outputs a negative log probability tensor with shape `(n_chains,)`. For HMC and MALA kernels, the function should be differentiable according to PyTorch autodiff.

We support various NF architectures like Real NVP, RQ-NSF, continuous NFs, ResFlow.
This package depends on [torchflows](https://github.com/davidnabergoj/torchflows) for NF definitions.
Please implement custom NF architectures in torchflows for compatibility with this package.
Example negative log probability density functions are provided in the accompanying [potentials package](https://github.com/davidnabergoj/potentials).

An example using Real NVP and a standard Gaussian target is shown below.

```python
import torch
from nfmc import sample

torch.manual_seed(0)  # Set the random seed for reproducible results

n_iterations = 1000
n_chains = 100
n_dim = 25  # Each event is a vector with size 25.


# Define the target negative log probability density
def neg_log_prob_target(x):
    return torch.sum(x ** 2, dim=-1)


# Draw samples with Ex2 HMC sampler
draws = sample(
    neg_log_prob_target=neg_log_prob_target,
    event_shape=(n_dim,),
    kernel="ex2_hmc",
    flow="realnvp",
    n_chains=n_chains,
    n_sampling_steps=5000,
    warmup=True
)
```

The `draws` output is a `Samples` object, which stores observed chain states, as well as their first and second moments.

### Specifying the MCMC kernel

The default kernel is IMH.
To use different kernels, pass the following keyword arguments to the `sample` function:

```python
# Independent Metropolis-Hastings
sample(..., kernel='imh')

# Local Metropolis-Hastings with occassional global NF proposals (jumps)
sample(..., kernel='jump_rwmh')  # RWMH
sample(..., kernel='jump_mala')  # MALA
sample(..., kernel='jump_hmc')   # HMC

# Local Metropolis-Hastings with occassional global NF proposals (jumps, multiple candidates)
sample(..., kernel='ex2_rwmh')  # RWMH
sample(..., kernel='ex2_mala')  # MALA
sample(..., kernel='ex2_hmc')   # HMC

# Preconditioned local Metropolis-Hastings
sample(..., kernel='neutra_rwmh')  # RWMH
sample(..., kernel='neutra_mala')  # MALA
sample(..., kernel='neutra_hmc')   # HMC
```

### Specifying the normalizing flow

The default NF is Real NVP. To use different kernels, pass the following keyword arguments to the `sample` function:
```python
# Fast architectures
sample(..., flow='nice')       # Dinh et al. (2015)
sample(..., flow='realnvp')    # Dinh et al. (2016)
sample(..., flow='c-rqnsf')    # Durkan et al. (2019)
sample(..., flow='c-lrsnsf')   # Dolatabadi et al. (2020)

# Potentially slower architectures (linear time scaling with target dimensionality for certain kernels)
sample(..., flow='iaf')        # Kingma et al. (2017)
sample(..., flow='maf')        # Papamakarios et al. (2018)
sample(..., flow='ma-rqnsf')   # Durkan et al. (2019)
sample(..., flow='ia-rqnsf')   # Durkan et al. (2019)

# Potentially slower architectures (ODE or numerical simulation)
sample(..., flow='i-resnet')   # Behrmann et al. (2019)
sample(..., flow='resflow')    # Chen et al. (2020)
sample(..., flow='ffjord')     # Gratwohl et al. (2018)
sample(..., flow='ot-flow')    # Onken et al. (2021)

# Convolutional architectures for distributions where events are images
sample(..., flow='glow-nice')
sample(..., flow='glow-realnvp')
sample(..., flow='glow-rqnsf')
sample(..., flow='glow-lrsnsf')
sample(..., flow='conv-i-resnet')
sample(..., flow='conv-resflow')
sample(..., flow='conv-ffjord')
```

To specify hyperparameters in NF objects, create them according to the torchflows package:
```python
from torchflows.flows import Flow
from torchflows.architectures import RealNVP

event_shape = (100,)  # Set according to target distribution

# Use a custom number of layers
bijection = RealNVP(event_shape, n_layers=5)
flow = Flow(bijection)
sample(..., flow=flow)
```
See [torchflows](github.com/davidnabergoj/torchflows) for more information on creating NF objects.

### Creating sampler objects manually

The previously described `sample` function allows straightforward sampling and warmup for different samplers.
The user may also manually create a sampler object.

Other samplers may be used by explicitly creating the sampler object and calling the `sample` method.
We provide an example for NeuTra HMC and the standard Gaussian target distribution:

```python
import torch
from torchflows.flows import Flow
from torchflows.architectures import RealNVP
from nfmc.algorithms.mh.preconditioning.samplers.neutra import NeuTraHMC

torch.manual_seed(0)

event_shape = (7,)
n_chains = 80

def neg_log_prob_target(x):
    return torch.sum(x ** 2, dim=-1)

# Generate initial latent states
z_initial = torch.rand(size=(n_chains, *event_shape)) * 4 - 2

# Create the flow object
bijection = RealNVP(event_shape, n_layers=5)
flow = Flow(bijection)

# Create the sampler object
sampler = NeuTraHMC(
    flow=flow,
    neg_log_prob_target=neg_log_prob_target
)

# Run warmup to tune kernel parameters and NF preconditioner
_, latent_warmup_draws = sampler.warmup(
    z0=z_initial,
    n_cycles=5,
    cycle_length=1000,
    return_latent_samples=True
)

# Sample with fixed kernel
target_draws = sampler.sample(
    z0=latent_warmup_draws.last_sample,
    n_steps=1000
)

print(target_draws.as_tensor().shape)  # (1000, 80, 7)
```

## Contributing

We warmly welcome any contributions or comments.
Some aspects of the package that can be improved:

* Additional sampler tests.
* Implementation of PMC, NS, SNF, AFT, CRAFT, FAB and other NFMC methods.
* Configuring a continuous integration pipeline with Github actions.
