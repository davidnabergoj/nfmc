from nfmc.algorithms.kernel import MarkovKernel
from nfmc.algorithms.base.sampler import MCMCSampler
from nfmc.algorithms.util.samples import Samples


import torch
from tqdm import tqdm


import time
from copy import deepcopy
from typing import List, Tuple, Union


class PreconditionedMCMCSampler(MCMCSampler):
    """
    Sampler class for MCMC algorithms with preconditioning.

    All kernel transitions are performed according to a preconditioner-adjusted target density.
    """

    def __init__(self,
                 kernel: MarkovKernel,
                 extra_warmup_kernels: List[Tuple[MarkovKernel, int]] = None,
                 **kwargs):
        """Preconditioned MCMC sampler constructor.

        :param MarkovKernel kernel: MCMC kernel to use.
        :param List[Tuple[MarkovKernel, int]] warmup_kernel_sequence: sequence of tuples. Each tuple
         contains a kernel object and an integer representing the number of cycles. The kernel is
         used for this many cycles during warmup. After the last of the extra warmup kernels is 
         used, the main kernel is used for the remainder of warmup.
        """
        if extra_warmup_kernels is None:
            extra_warmup_kernels = []
        # Convert to list of lists
        self.extra_warmup_kernels = [[x[0], x[1]]
                                     for x in extra_warmup_kernels]
        super().__init__(kernel, **kwargs)

    @property
    def name(self) -> str:
        return "Generic preconditioned MCMC sampler"

    @property
    def active_warmup_kernel(self):
        if len(self.extra_warmup_kernels) > 0:
            return self.extra_warmup_kernels[0][0]
        else:
            return self.kernel

    def advance_warmup_kernel(self) -> MarkovKernel:
        """
        Advance to the next warmup kernel.
        """
        if self.extra_warmup_kernels:
            if self.extra_warmup_kernels[0][1] > 0:
                self.extra_warmup_kernels[0][1] -= 1

            if self.extra_warmup_kernels[0][1] == 0:
                self.extra_warmup_kernels.pop(0)

    def warmup(self,
               z0: torch.Tensor,
               n_cycles: int,
               cycle_length: int,
               show_progress: bool = True,
               show_fit_progress: bool = False,
               time_limit_seconds: Union[float, int] = None,
               max_samples: int = None,
               max_training_samples: int = None,
               data_transform: callable = None,
               return_latent_samples: bool = False,
               **kwargs) -> Union[Samples, Tuple[Samples, Samples]]:
        """
        Optimize kernel parameters.

        The kernel's preconditioner is updated every cycle.
        The kernel's preconditioner is not updated within the final cycle so that the rest of the kernel can be stably 
         tuned.
        The kernel is updated in each step within the first half of a cycle unless it internally overrides this.
        There are no kernel updates in the last half-cycle, which is instead reserved for burn-in.

        :param torch.Tensor z0: initial latent state with shape `(*batch_shape, *event_shape)`.
        :param int n_cycles: number of warmup cycles.
        :param int cycle_length: number of MCMC steps in each warmup cycle.
        :param bool show_progress: if True, display a progress bar.
        :param float time_limit_seconds: maximum sampling time. Sampling stops if this time is exceeded.
        :param int max_samples: maximum number of samples to store.
        :param int max_training_samples: maximum number of training samples to train the preconditioner.
        :param callable data_transform: function that transforms each generated sample. Receives as input a tensor with
         shape `(*batch_shape, *event_shape)` and outputs a tensor with shape `(*batch_shape, *event_shape)`.
        :param bool return_latent_samples: if True, return tuple with two Samples objects. The first object holds samples
         from the target distribution, the second holds latent samples. The specified data_transform callable is still
         applied to samples in each object.
        :return: Samples object with MCMC draws.
        """
        self.kernel.start_warmup(n_chains=len(z0))
        for i in range(len(self.extra_warmup_kernels)):
            self.extra_warmup_kernels[i][0].start_warmup(n_chains=len(z0))

        target_samples = Samples(
            event_shape=self.kernel.event_shape,
            max_samples=max_samples,
            data_transform=data_transform
        )

        latent_samples = Samples(
            event_shape=self.kernel.event_shape,
            max_samples=max_samples,
        )

        _adj_max = max_training_samples
        if max_training_samples is not None:
            _adj_max /= len(z0)  # divide by number of chains
        training_samples = Samples(
            event_shape=self.kernel.event_shape,
            max_samples=_adj_max
        )

        self.kernel.reset_statistics()
        z = deepcopy(z0.detach())

        t0 = time.time()
        pbar = tqdm(
            range(n_cycles * cycle_length),
            desc=f'Warmup',
            disable=not show_progress
        )

        def resample(states, method: str):
            """
            Resample states according to method.

            :param torch.Tensor states: tensor with shape `(n_steps, n_chains, *event_shape)`.
            :param str method: resampling method. One of 'uniform', 'divergence', 'density'.
            :return: resampled states with shape `(n_steps, n_chains, *event_shape)`.
            """
            if len(states.shape) != 2 + len(self.kernel.event_shape):
                raise ValueError(
                    f'Expected states to have shape (n_steps, n_chains, *event_shape), got {states.shape = }'
                )
            n_steps, n_chains = states.shape[:2]

            if method == 'uniform':
                flat_states = states.flatten(0, 1)
                resampled_flat_states = flat_states[
                    torch.randint(len(flat_states), (len(flat_states),), device=flat_states.device)
                ]
                return resampled_flat_states.view_as(states)
            else:
                negative_log_weights: torch.Tensor  # (n_steps, n_chains)
                if method == 'divergence':
                    if isinstance(self.kernel._n_divergences_per_chain, int):
                        # Fall-back
                        return resample(states, method='uniform')
                    negative_log_weights = self.kernel._n_divergences_per_chain.to(
                        dtype=states.dtype,
                        device=states.device
                    )  # (n_chains,)
                    negative_log_weights = negative_log_weights[None].repeat(n_steps, 1)
                    _s = negative_log_weights.std()
                    if _s == 0:
                        negative_log_weights = negative_log_weights - negative_log_weights.mean()
                    else:
                        negative_log_weights = torch.divide(
                            negative_log_weights - negative_log_weights.mean(),
                            negative_log_weights.std() + 1e-10
                        )
                elif method == 'density':
                    negative_log_weights = []
                    for chain_id in range(len(states)):
                        try:
                            negative_log_weights.append(
                                current_warmup_kernel.neg_log_prob_target(
                                    states[chain_id].unsqueeze(0)
                                )
                            )
                        except ValueError:
                            negative_log_weights.append(
                                torch.tensor([torch.inf], device=states.device)
                            )
                    negative_log_weights = torch.cat(negative_log_weights, dim=0)
                else:
                    raise ValueError(f'Unknown resampling method {method}')
                
                if negative_log_weights.shape != (n_steps, n_chains):
                    raise ValueError(
                        f'Expected negative_weights.shape to be {(n_steps, n_chains) = }, got {negative_log_weights.shape = }'
                    )
                log_weights = -negative_log_weights
                log_weights_flat = log_weights.flatten()

                # Replace invalid values with -inf (so they get zero probability)
                log_weights_flat = torch.where(
                    torch.isfinite(log_weights_flat),
                    log_weights_flat,
                    -torch.inf
                )
                probabilities = torch.softmax(log_weights_flat, dim=0)
                # If all probabilities are NaN/zero, fall back to uniform
                if not torch.isfinite(probabilities).any():
                    probabilities = torch.ones_like(
                        probabilities
                    ) / len(probabilities)

                indices = torch.multinomial(
                    probabilities, 
                    num_samples=n_steps * n_chains, 
                    replacement=True
                )
                return states.flatten(0, 1)[indices].view_as(states)

        for cycle_index in range(n_cycles):
            z = z.detach().clone()
            if not torch.isfinite(z).all():
                raise ValueError("Initial state has nan/inf values")

            current_warmup_kernel = self.active_warmup_kernel

            if cycle_index > 0:
                # Transform current latent state to target space
                # This could be unstable
                x, _ = current_warmup_kernel._preconditioner.inverse_transform(
                    z.clone()
                )
                if not torch.isfinite(x).all():
                    raise ValueError("Preconditioner inverse returned nan/inf values")
                # Resample target states, hopefully getting rid of stuck chains over time
                x = resample(x[None].clone(), method='divergence')[0]

                self.advance_warmup_kernel()
                current_warmup_kernel = self.active_warmup_kernel

                # Update the preconditioner first so drawn sample can contribute toward next preconditioner fit.

                x_train = training_samples.as_tensor()  # Convert training data to torch.Tensor
                x_train = resample(x_train.clone(), method='divergence')

                # Flatten steps and chains
                x_train = x_train.view(-1, *current_warmup_kernel.event_shape)

                # Remove nan/inf training data
                x_train = x_train[
                    torch.isfinite(x_train).all(dim=tuple(range(1, x_train.ndim)))
                ]
                if torch.numel(x_train) == 0:
                    raise ValueError("Got zero training data points")
                current_warmup_kernel.fit_preconditioner(
                    x_train.clone(),
                    show_progress=show_fit_progress,
                    **kwargs
                )

                # Reset state and kernel
                z, _ = current_warmup_kernel._preconditioner.forward_transform(
                    x.clone()
                )

                # Mask invalid states (NaN or Inf anywhere along dimensions beyond the batch)
                invalid_mask = ~torch.all(torch.isfinite(z), dim=tuple(range(1, z.ndim)))
                valid_mask = ~invalid_mask

                # Indices of valid states
                valid_indices = valid_mask.nonzero(as_tuple=True)[0]

                # Replace invalid states by sampling from valid states
                if valid_indices.numel() > 0 and invalid_mask.any():
                    # Randomly choose replacement indices from valid ones
                    replacement_indices = torch.randint(
                        0, valid_indices.numel(), size=(invalid_mask.sum(),), device=z.device
                    )
                    z[invalid_mask] = z[valid_indices[replacement_indices]]
                
                if not torch.isfinite(z).all():
                    raise ValueError("Preconditioner forward returned nan/inf values")

                current_warmup_kernel.reset_parameters()
            
            for step_index in range(cycle_length):
                # Step. Update if in first half of cycle.
                do_update = step_index < cycle_length // 2
                if step_index == (cycle_length // 2):
                    pass

                z, x = current_warmup_kernel.step_with_preconditioner_inverse(
                    z.clone(),
                    update=do_update
                )
                if not torch.isfinite(z).all():
                    raise ValueError("Kernel step returned nan/inf values")
                if not torch.isfinite(x).all():
                    raise ValueError("Kernel step preconditioner inverse returned nan/inf values")

                if not do_update:
                    training_candidates = x.detach().clone().view(-1, *self.kernel.event_shape)
                    training_samples.add(training_candidates)

                target_samples.add(x.detach().clone())
                if return_latent_samples:
                    latent_samples.add(z.detach().clone())

                elapsed_time = time.time() - t0
                pbar.set_postfix_str(
                    current_warmup_kernel.pbar_repr(elapsed_time)
                )
                pbar.update(1)
                pbar.refresh()
                if time_limit_seconds is not None and elapsed_time > time_limit_seconds:
                    break

        self.kernel.end_warmup()
        self.kernel.finalize_parameters()

        if return_latent_samples:
            return target_samples, latent_samples
        return target_samples

    def sample(self,
               z0: torch.Tensor,
               n_steps: int,
               show_progress: bool = True,
               time_limit_seconds: Union[float, int] = None,
               max_samples: int = None,
               data_transform: callable = None,
               return_latent_samples: bool = False) -> Samples:
        """
        Sample with a fixed kernel.

        The kernel is updated every step unless it internally overrides this.
        The kernel's preconditioner is updated every K steps where K is equal to preconditioner_update_interval.

        :param torch.Tensor z0: initial latent state with shape `(*batch_shape, *event_shape)`.
        :param int n_steps: number of MCMC steps to perform.
        :param bool show_progress: if True, display a progress bar.
        :param float time_limit_seconds: maximum sampling time. Sampling stops if this time is exceeded.
        :param int max_samples: maximum number of samples to store.
        :param callable data_transform: function that transforms each generated sample. Receives as input a tensor with
         shape `(*batch_shape, *event_shape)` and outputs a tensor with shape `(*batch_shape, *event_shape)`.
        :param bool return_latent_samples: if True, return tuple with two Samples object. The first object holds samples
         from the target distribution, the second holds latent samples. The specified data_transform callable is still 
         applied to samples in each object.
        """
        target_samples = Samples(
            event_shape=self.kernel.event_shape,
            max_samples=max_samples,
            data_transform=data_transform
        )

        latent_samples = Samples(
            event_shape=self.kernel.event_shape,
            max_samples=max_samples,
        )

        self.kernel.reset_statistics()
        z = deepcopy(z0.detach())

        t0 = time.time()
        for _ in (pbar := tqdm(range(n_steps),
                               desc=f'Sampling',
                               disable=not show_progress)):
            z = z.detach().clone()
            z, x = self.kernel.step_with_preconditioner_inverse(z.clone())

            target_samples.add(x.detach().clone())
            if return_latent_samples:
                latent_samples.add(z.detach().clone())

            elapsed_time = time.time() - t0
            pbar.set_postfix_str(self.kernel.pbar_repr(elapsed_time))
            if time_limit_seconds is not None and elapsed_time > time_limit_seconds:
                break

        if return_latent_samples:
            return target_samples, latent_samples
        return target_samples
