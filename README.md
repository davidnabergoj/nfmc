# Normalizing flow Monte Carlo

This package provides various Markov chain Monte Carlo (MCMC) algorithms that utilize normalizing flows (NF).
These are listed in the following table.
The last column specifies whether the algorithm relies on both forward and inverse maps of the underlying NF bijection
or only one.

| Sampler                                                | Abbreviation | Requires NF forward and inverse | Requires prior | 
|--------------------------------------------------------|:------------:|:-------------------------------:|:--------------:|
| Independent Metropolis Hastings                        |     IMH      |               Yes               |       No       |
| Metropolis adjusted Langevin algorithm                 |   NF-MALA    |               Yes               |       No       |
| Unadjusted Langevin algorithm                          |    NF-ULA    |               Yes               |       No       |
| NeuTra HMC                                             |    NeuTra    |               No                |       No       |
| Transport elliptical slice sampling                    |     TESS     |               Yes               |       No       |
| Nested sampling                                        |    NESSAI    |               Yes               |      Yes       |
| Deterministic Langevin Monte Carlo                     |     DLMC     |               *No               |      Yes       |
| Stochastic normalizing flows                           |     SNF      |               No                |      Yes       |
| Preconditioned Monte Carlo                             |     PMC      |               Yes               |      Yes       |
| Annealed flow transport Monte Carlo                    |     AFT      |               No                |      Yes       |
| Continual repeated annealed flow transport Monte Carlo |    CRAFT     |               No                |      Yes       |
| Flow annealed importance sampling bootstrap            |     FAB      |               Yes               |       No       |

&ast; Standard DLMC only requires the forward map, but latent DLMC requires the inverse map as well. 
