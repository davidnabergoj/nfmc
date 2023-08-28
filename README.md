# Normalizing flow Monte Carlo

This package provides various Markov chain Monte Carlo (MCMC) algorithms that utilize normalizing flows (NF).
These are listed in the following table.
The last column specifies whether the algorithm relies on both forward and inverse maps of the underlying NF bijection
or only one.

| Sampler                                                | Abbreviation |   Kind    | Requires NF forward and inverse | Requires prior | 
|--------------------------------------------------------|:------------:|:---------:|:-------------------------------:|:--------------:|
| Independent Metropolis Hastings                        |     IMH      | Sampling  |               Yes               |       No       |
| Metropolis adjusted Langevin algorithm                 |   NF-MALA    | Sampling  |               Yes               |       No       |
| Unadjusted Langevin algorithm                          |    NF-ULA    | Sampling  |               Yes               |       No       |
| NeuTra HMC                                             |    NeuTra    | Sampling  |               No                |       No       |
| Transport elliptical slice sampling                    |     TESS     | Sampling  |               Yes               |       No       |
| Nested sampling                                        |    NESSAI    | Transport |               Yes               |      Yes       |
| Deterministic Langevin Monte Carlo                     |     DLMC     | Transport |               *No               |      Yes       |
| Stochastic normalizing flows                           |     SNF      | Transport |               No                |      Yes       |
| Preconditioned Monte Carlo                             |     PMC      | Transport |               Yes               |      Yes       |
| Annealed flow transport Monte Carlo                    |     AFT      | Transport |               No                |      Yes       |
| Continual repeated annealed flow transport Monte Carlo |    CRAFT     | Transport |               No                |      Yes       |
| Flow annealed importance sampling bootstrap            |     FAB      | Transport |               Yes               |       No       |

&ast; Standard DLMC only requires the forward map, but latent DLMC requires the inverse map as well. 
