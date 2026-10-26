# Markov Chain Monte Carlo with Normalizing Flows

This package implements MCMC algorithms that use normalizing flows (NFs).
It supports the following Markov kernels:

* Independent Metropolis-Hastings (IMH)
* Iterated sampling importance resampling (i-SIR)
* Random walk Metropolis-Hastings (RWMH)
* Metropolis adjusted Langevin algorithm (MALA)
* Hamiltonian Monte Carlo (HMC)
* No-U-turn sampler (NUTS)

NFs can be used as preconditioners, as in the NeuTra MCMC framework (Hoffman et al., 2019).
They can also be used as independent proposal distributions, as in IMH, or combined with local kernels that make occasional global NF jumps, as in Ex2 MCMC (Samsonov et al., 2022).
Diagonal and dense (non-diagonal) linear preconditioners are available as alternatives to NFs.
The samplers are implemented in PyTorch.

This package is intended for research purposes or for sampling when the time cost of computing the target probability density is much greater than the time cost of NF operations.

## Installation

This package requires Python 3.8+ (CI tests run on Python 3.9) and depends on [torchflows](https://github.com/davidnabergoj/torchflows) (version 1.2.1 or newer).

Install the package directly from GitHub:
```
pip install git+https://github.com/davidnabergoj/nfmc.git
```

To configure the package for local development, clone the repository and install dependencies:
```
git clone git@github.com:davidnabergoj/nfmc.git
cd nfmc
pip install -r requirements.txt
```

Continuous NF architectures (e.g., FFJORD, OT-Flow, RNODE) additionally require `torchdiffeq`, which is included in `requirements.txt`.

Run the test suite with:
```
pytest -m "not local_only"
```

## Sampling example

Each target distribution is defined by a function that computes its (unnormalized) negative log probability density.
The function accepts a tensor with shape `(n_chains, *event_shape)` and outputs a tensor with shape `(n_chains,)`.
For kernels that use gradients (MALA, HMC, NUTS and their variants), the function must be differentiable with PyTorch autodiff.
Example negative log probability density functions are provided in the accompanying [potentials package](https://github.com/davidnabergoj/potentials).
If the target object has an `event_shape` attribute, the `event_shape` argument can be omitted.

An example using Ex2 HMC with a Real NVP flow and a standard Gaussian target is shown below.

```python
import torch
from nfmc import sample

torch.manual_seed(0)  # Set the random seed for reproducible results

n_dim = 25  # Each event is a vector with size 25


# Define the target negative log probability density
def neg_log_prob_target(x):
    return torch.sum(x ** 2, dim=-1) / 2


draws = sample(
    neg_log_prob_target=neg_log_prob_target,
    event_shape=(n_dim,),
    kernel="ex2_hmc",
    flow="realnvp",
    n_chains=100,
    n_sampling_steps=1000,
    warmup=True,
)

print(draws.as_tensor().shape)  # torch.Size([1000, 100, 25])
```

The `sample` function returns a `Samples` object (`nfmc.algorithms.util.samples.Samples`) with the following attributes:

* `draws.as_tensor()`: stored draws with shape `(n_steps, n_chains, *event_shape)`,
* `draws.n_samples`: number of stored steps,
* `draws.last_sample`: final chain states with shape `(n_chains, *event_shape)`,
* `draws.first_moment.as_tensor()` and `draws.second_moment.as_tensor()`: running estimates of `E[x]` and `E[x^2]`, computed over all draws.

### Main `sample` arguments

```python
sample(
    neg_log_prob_target,               # callable: (n_chains, *event_shape) -> (n_chains,)
    event_shape=None,                  # tuple; inferred from the flow or target if available
    flow='realnvp',                    # NF name, Flow object, or None (for standard MCMC kernels)
    kernel='imh',                      # see "Specifying the MCMC kernel"
    n_sampling_steps=100,              # number of MCMC steps after warmup
    n_chains=100,                      # ignored if x0 is provided
    x0=None,                           # initial states with shape (n_chains, *event_shape); defaults to Uniform(-2, 2)
    warmup=False,                      # if True, tune the kernel (and fit the preconditioner) before sampling
    warmup_kwargs=None,                # passed to sampler.warmup (see below)
    sample_kwargs=None,                # passed to sampler.sample, e.g., {'max_samples': 500}
    show_progress=True,
    sampling_time_limit_seconds=None,
    warmup_time_limit_seconds=None,
    return_warmup_samples=False,       # if True, return (sampling_draws, warmup_draws)
    **kwargs                           # passed to create_sampler: kernel_kwargs, sampler_kwargs, flow_kwargs, device
)
```

For NF-based kernels, warmup consists of `n_cycles` cycles (default 5) with `cycle_length` steps each (default 2000).
Kernel parameters (e.g., step sizes) are tuned in the first half of each cycle, and states from the second half are stored as training data.
The preconditioner is fitted to this training data at the start of every cycle after the first one.
The `max_training_samples` warmup argument limits the size of the training set, and any remaining keyword arguments (e.g., `n_epochs`, `lr`, `batch_size`) are passed to the flow's `fit` method:

```python
sample(
    ...,
    warmup=True,
    warmup_kwargs={
        'n_cycles': 10,
        'cycle_length': 500,
        'max_training_samples': 10_000,
        'n_epochs': 200,  # passed to Flow.fit
    }
)
```

For standard MCMC kernels (`rwmh`, `mala`, `hmc`, `nuts`), warmup is a single tuning run with `n_steps` steps (defaults to `n_sampling_steps`):

```python
sample(..., kernel='hmc', flow=None, warmup=True, warmup_kwargs={'n_steps': 500})
```

Kernel hyperparameters are set with `kernel_kwargs`, for example:

```python
sample(..., kernel='neutra_hmc', kernel_kwargs={'n_leapfrog_steps': 10, 'step_size': 0.05})
```

Note that for `jump_*` and `ex2_*` kernels, `kernel_kwargs` are passed to both the local and the global kernel.

### Specifying the MCMC kernel

The default kernel is IMH.
To use different kernels, pass the following keyword arguments to the `sample` function:

```python
# Standard MCMC (no NF)
sample(..., kernel='rwmh')
sample(..., kernel='mala')
sample(..., kernel='hmc')
sample(..., kernel='nuts')

# Independent proposals from the NF
sample(..., kernel='imh')    # Independent Metropolis-Hastings
sample(..., kernel='i-sir')  # Iterated sampling importance resampling

# Local kernels with occasional global NF proposals (jumps with IMH)
sample(..., kernel='jump_rwmh')
sample(..., kernel='jump_mala')
sample(..., kernel='jump_hmc')
sample(..., kernel='jump_nuts')

# Local kernels with occasional global NF proposals (jumps with i-SIR, multiple candidates; Ex2 MCMC)
sample(..., kernel='ex2_rwmh')
sample(..., kernel='ex2_mala')
sample(..., kernel='ex2_hmc')
sample(..., kernel='ex2_nuts')

# NF-preconditioned local kernels (NeuTra MCMC)
sample(..., kernel='neutra_rwmh')
sample(..., kernel='neutra_mala')
sample(..., kernel='neutra_hmc')
sample(..., kernel='neutra_nuts')
```

In the `jump_*` and `ex2_*` kernels, local transitions are also preconditioned with the NF.

### Specifying the normalizing flow

The default NF is Real NVP. To use a different architecture, pass its name to the `sample` function:
```python
# Fast architectures
sample(..., flow='nice')          # Dinh et al. (2015)
sample(..., flow='realnvp')       # Dinh et al. (2017)
sample(..., flow='c-rqnsf')       # Durkan et al. (2019)
sample(..., flow='c-lrsnsf')      # Dolatabadi et al. (2020)
sample(..., flow='c-naf-deep')    # Huang et al. (2018)

# Masked autoregressive architectures (linear time scaling with target dimensionality in one direction)
sample(..., flow='maf')           # Papamakarios et al. (2017)
sample(..., flow='iaf')           # Kingma et al. (2016)
sample(..., flow='ma-rqnsf')      # Durkan et al. (2019)
sample(..., flow='ia-rqnsf')      # Durkan et al. (2019)
sample(..., flow='ma-lrsnsf')
sample(..., flow='ia-lrsnsf')

# Residual architectures
sample(..., flow='i-resnet')          # Behrmann et al. (2019)
sample(..., flow='resflow')           # Chen et al. (2019)
sample(..., flow='proximal-resflow')  # Hertrich (2023)
sample(..., flow='planar')            # Rezende & Mohamed (2015)
sample(..., flow='radial')            # Rezende & Mohamed (2015)
sample(..., flow='sylvester')         # van den Berg et al. (2018)

# Continuous architectures (ODE-based, potentially slow)
sample(..., flow='ffjord')        # Grathwohl et al. (2019)
sample(..., flow='rnode')         # Finlay et al. (2020)
sample(..., flow='ot-flow')       # Onken et al. (2021)
sample(..., flow='ddb')           # Salman et al. (2018)

# Multiscale and convolutional architectures for distributions where events are images
sample(..., flow='ms-realnvp')
sample(..., flow='glow-realnvp')
sample(..., flow='glow-nice')
sample(..., flow='glow-rqnsf')
sample(..., flow='glow-lrsnsf')
sample(..., flow='conv-i-resnet')
sample(..., flow='conv-resflow')
sample(..., flow='conv-ffjord')
```

The full list of supported names (including synonyms) is given by:
```python
from nfmc.util import get_supported_normalizing_flows

print(get_supported_normalizing_flows(synonyms=False))
```

Flow hyperparameters can be passed with `flow_kwargs` or encoded in the flow string as `<name>%<json>`:
```python
sample(..., flow='realnvp', flow_kwargs={'n_layers': 5})
sample(..., flow='realnvp%{"n_layers": 5}')
```

Alternatively, create the NF object according to the torchflows package and pass it directly.
In this case, `event_shape` is taken from the flow:
```python
from torchflows.flows import Flow
from torchflows.bijections.finite.autoregressive.architectures import RealNVP

event_shape = (100,)  # Set according to the target distribution

flow = Flow(RealNVP(event_shape, n_layers=5))
sample(..., flow=flow)
```
See [torchflows](https://github.com/davidnabergoj/torchflows) for more information on creating NF objects.
Custom NF architectures should be implemented in torchflows for compatibility with this package.

### Creating sampler objects manually

The `sample` function builds the sampler with `nfmc.sample.create_sampler`, which takes the same `flow`, `kernel`, `kernel_kwargs`, `sampler_kwargs`, `flow_kwargs` and `device` arguments and returns a sampler object.
Sampler classes can also be created directly and used via their `warmup` and `sample` methods.
For preconditioned samplers, both methods operate on latent states `z`, while the returned `Samples` objects contain target-space draws (pass `return_latent_samples=True` to also obtain latent draws).

We provide an example for NeuTra HMC and a standard Gaussian target distribution:

```python
import torch
from torchflows.flows import Flow
from torchflows.bijections.finite.autoregressive.architectures import RealNVP
from nfmc.algorithms.preconditioning.samplers.neutra import NeuTraHMC

torch.manual_seed(0)

event_shape = (7,)
n_chains = 80


def neg_log_prob_target(x):
    return torch.sum(x ** 2, dim=-1) / 2


# Generate initial latent states
z_initial = torch.rand(size=(n_chains, *event_shape)) * 4 - 2

# Create the flow object
flow = Flow(RealNVP(event_shape, n_layers=5))

# Create the sampler object
sampler = NeuTraHMC(
    flow=flow,
    neg_log_prob_target=neg_log_prob_target
)

# Run warmup to tune kernel parameters and the NF preconditioner
_, latent_warmup_draws = sampler.warmup(
    z0=z_initial,
    n_cycles=5,
    cycle_length=1000,
    return_latent_samples=True
)

# Sample with a fixed kernel
target_draws = sampler.sample(
    z0=latent_warmup_draws.last_sample,
    n_steps=1000
)

print(target_draws.as_tensor().shape)  # torch.Size([1000, 80, 7])
```

Other available sampler classes include:

* `nfmc.algorithms.preconditioning.samplers.neutra`: `NeuTraRWMH`, `NeuTraMALA`, `NeuTraHMC`,
* `nfmc.algorithms.preconditioning.samplers.diagonal`: `DiagonalRWMH`, `DiagonalMALA`, `DiagonalHMC` (constructed with `event_shape` instead of `flow`),
* `nfmc.algorithms.preconditioning.samplers.dense`: `DenseRWMH`, `DenseMALA`, `DenseHMC` (constructed with `event_shape` instead of `flow`),
* `nfmc.algorithms.jump.samplers`: `NeuTraJumpRWMH`, `NeuTraJumpMALA`, `NeuTraJumpHMC` (NF-preconditioned local kernel) and `DiagonalJumpRWMH`, `DiagonalJumpMALA`, `DiagonalJumpHMC` (diagonally preconditioned local kernel), with `global_kernel='imh'` or `global_kernel='i-sir'`,
* `nfmc.algorithms.mh.base.MHSampler`: unpreconditioned sampler that wraps a kernel such as `HMCKernel` (`nfmc.algorithms.mh.local.hmc`) or `NUTSKernel` (`nfmc.algorithms.nuts`); its `warmup` and `sample` methods take target-space states `x0`.

## Contributing

We warmly welcome any contributions or comments.
Some aspects of the package that can be improved:

* Additional sampler tests.
* Implementation of PMC, NS, SNF, AFT, CRAFT, FAB and other NFMC methods.
