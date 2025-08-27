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
               time_limit_seconds: Union[float, int] = None,
               max_samples: int = None,
               max_training_samples: int = None,
               data_transform: callable = None,
               return_latent_samples: bool = False,
               **kwargs) -> Samples:
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
        :param kwargs: keyword arguments for `preconditioner.fit`.
        :return: Samples object with MCMC draws.
        """
        self.kernel.start_warmup(n_chains=len(z0))

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
            max_samples=_adj_max,
        )

        self.kernel.reset_statistics()
        z = deepcopy(z0.detach())

        t0 = time.time()
        pbar = tqdm(
            range(n_cycles * cycle_length),
            desc=f'Warmup',
            disable=not show_progress
        )

        def resample(states):
            neg_log_prob_states = []
            for chain_id in range(len(states)):
                try:
                    neg_log_prob_states.append(
                        current_warmup_kernel.neg_log_prob_target(
                            states[chain_id].unsqueeze(0)
                        )
                    )
                except ValueError:
                    neg_log_prob_states.append(
                        torch.tensor([torch.inf], device=states.device)
                    )
            neg_log_prob_states = torch.cat(neg_log_prob_states, dim=0)
            log_weights = -neg_log_prob_states

            # Replace invalid values with -inf (so they get zero probability)
            log_weights = torch.where(
                torch.isfinite(log_weights),
                log_weights,
                -torch.inf
            )
            probabilities = torch.softmax(log_weights, dim=0)
            # If all probabilities are NaN/zero, fall back to uniform
            if not torch.isfinite(probabilities).any():
                probabilities = torch.ones_like(
                    probabilities
                ) / len(probabilities)

            indices = torch.multinomial(
                probabilities, 
                num_samples=len(states), 
                replacement=True
            )
            return states[indices]

        for cycle_index in range(n_cycles):
            z = z.detach().clone()

            current_warmup_kernel = self.active_warmup_kernel
            current_warmup_kernel.start_warmup(n_chains=len(z0))

            if cycle_index > 0:
                # Transform current latent state to target space
                x, _ = current_warmup_kernel._preconditioner.inverse_transform(
                    z.clone()
                )

                # Resample target states according to the target log probability density
                # This gets rid of stuck chains
                x = resample(x.clone())

                self.advance_warmup_kernel()
                current_warmup_kernel = self.active_warmup_kernel
                current_warmup_kernel.start_warmup(n_chains=len(z0))

                # Update the preconditioner first so drawn sample can contribute toward next preconditioner fit.

                x_train = training_samples.as_tensor()  # Convert training data to torch.Tensor
                # Flatten steps and chains
                x_train = x_train.view(-1, *current_warmup_kernel.event_shape)
                if torch.numel(x_train) == 0:
                    raise ValueError("Got zero training data points")
                current_warmup_kernel.fit_preconditioner(
                    x_train.clone(), **kwargs)

                # Reset state and kernel
                z, _ = current_warmup_kernel._preconditioner.forward_transform(
                    x.clone())
                current_warmup_kernel.reset_parameters()
                training_samples = Samples(
                    event_shape=self.kernel.event_shape,
                    max_samples=_adj_max,
                )
            
            for step_index in range(cycle_length):
                # Step. Update if in first half of cycle.
                do_update = step_index < cycle_length // 2
                if step_index == (cycle_length // 2):
                    pass

                z, x = current_warmup_kernel.step_with_preconditioner_inverse(
                    z.clone(),
                    update=do_update
                )
                if not do_update:
                    training_samples.add(x.detach().clone())

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
