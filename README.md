# Normalizing flow Monte Carlo

This package provides various Markov chain Monte Carlo (MCMC) algorithms that utilize normalizing flows (NF).
These are listed in the following table.
The last column specifies whether the algorithm relies on both forward and inverse maps of the underlying NF bijection
or only one.

| Sampler                                                | Abbreviation | Requires NF forward and inverse |
|--------------------------------------------------------|:------------:|:-------------------------------:|
| Independent Metropolis Hastings                        |     IMH      |               Yes               |
| Metropolis adjusted Langevin algorithm                 |   NF-MALA    |               Yes               |
| Unadjusted Langevin algorithm                          |    NF-ULA    |               Yes               |
| NeuTra HMC                                             |    NeuTra    |               No                |
| Transport elliptical slice sampling                    |     TESS     |               Yes               |
| Nested sampling                                        |    NESSAI    |               Yes               |
| Deterministic Langevin Monte Carlo                     |     DLMC     |               *No               |
| Stochastic normalizing flows                           |     SNF      |               No                |
| Preconditioned Monte Carlo                             |     PMC      |               Yes               |
| Annealed flow transport Monte Carlo                    |     AFT      |               No                |
| Continual repeated annealed flow transport Monte Carlo |    CRAFT     |               No                |
| Flow annealed importance sampling bootstrap            |     FAB      |               Yes               |

&ast; Standard DLMC only requires the forward map, but latent DLMC requires the inverse map as well. 
