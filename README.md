# Normalizing flow Monte Carlo

This package provides various Markov chain Monte Carlo (MCMC) algorithms that utilize normalizing flows (NF):

| Sampler                                                | Abbreviation | Requires density estimation and sampling |
|--------------------------------------------------------|:------------:|:----------------------------------------:|
| Independent Metropolis Hastings                        |     IMH      |                   Yes                    |
| Metropolis adjusted Langevin algorithm                 |   NF-MALA    |                                          |
| Unadjusted Langevin algorithm                          |    NF-ULA    |                                          |
| NeuTra HMC                                             |    NeuTra    |                                          |
| Transport elliptical slice sampling                    |     TESS     |                                          |
| Nested sampling                                        |    NESSAI    |                                          |
| Deterministic Langevin Monte Carlo                     |     DLMC     |                                          |
| Stochastic normalizing flows                           |     SNF      |                                          |
| Preconditioned Monte Carlo                             |     PMC      |                                          |
| Annealed flow transport Monte Carlo                    |     AFT      |                                          |
| Continual repeated annealed flow transport Monte Carlo |    CRAFT     |                                          |
| Flow annealed importance sampling bootstrap            |     FAB      |                                          |
