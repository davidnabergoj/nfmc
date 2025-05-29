from nfmc.algorithms.kernel import MarkovKernel
from nfmc.algorithms.sampling.base.sampler import MCMCSampler
from nfmc.algorithms.util.samples import Samples


import torch
from tqdm import tqdm


import time
from copy import deepcopy
from typing import List, Union


class PreconditionedMCMCSampler(MCMCSampler):
    """
    Sampler class for MCMC algorithms with preconditioning.

    All kernel transitions are performed according to a preconditioner-adjusted target density.
    """

    @property
    def name(self) -> str:
        return "Generic preconditioned MCMC sampler"

    def prepare_training_data(self,
                              train_data_list: List[torch.Tensor],
                              max_training_samples: int):
        # Flatten training data elements
        train_data_list = [
            x.view(-1, *self.kernel.event_shape)
            for x in train_data_list
        ]
        x_train = torch.concat(train_data_list, dim=0)
        x_train = x_train[torch.randperm(len(x_train))]
        if max_training_samples is not None:
            x_train = x_train[:max_training_samples]
        return x_train

    def warmup(self,
               z0: torch.Tensor,
               n_steps: int,
               preconditioner_update_interval: int,
               show_progress: bool = True,
               time_limit_seconds: Union[float, int] = None,
               max_samples: int = None,
               max_training_samples: int = None,
               data_transform: callable = None,
               return_latent_samples: bool = False,
               **kwargs) -> Samples:
        """
        Optimize kernel parameters.

        The kernel is updated every step unless it internally overrides this.
        The kernel's preconditioner is updated every K steps where K is equal to preconditioner_update_interval.
        The kernel's preconditioner is not updated within the final K steps so that the rest of the kernel can be stably 
         tuned.

        :param torch.Tensor z0: initial latent state with shape `(*batch_shape, *event_shape)`.
        :param int n_steps: number of MCMC steps to perform.
        :param int preconditioner_update_interval: update the preconditioner after this number of MCMC steps.
        :param bool show_progress: if True, display a progress bar.
        :param float time_limit_seconds: maximum sampling time. Sampling stops if this time is exceeded.
        :param int max_samples: maximum number of samples to store.
        :param int max_training_samples: maximum number of training samples to train the preconditioner.
        :param callable data_transform: function that transforms each generated sample. Receives as input a tensor with
         shape `(*batch_shape, *event_shape)` and outputs a tensor with shape `(*batch_shape, *event_shape)`.
        :param bool return_latent_samples: if True, return tuple with two Samples object. The first object holds samples
         from the target distribution, the second holds latent samples. The specified data_transform callable is still 
         applied to samples in each object.
        :param kwargs: keyword arguments for `preconditioner.fit`.
        :return: Samples object with MCMC draws.
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

        # Holds data for preconditioner training/fitting
        x_train_list = []

        t0 = time.time()
        for step in (pbar := tqdm(range(n_steps),
                                  desc=f'Warmup',
                                  disable=not show_progress)):

            if step % preconditioner_update_interval == 0 and 0 < step < n_steps - preconditioner_update_interval:
                # Update the preconditioner first so drawn sample can contribute toward next preconditioner fit.
                with torch.no_grad():
                    x_train = self.prepare_training_data(
                        x_train_list,
                        max_training_samples
                    )
                self.kernel.fit_preconditioner(x_train, **kwargs)
                x_train_list = []

                # Reset state
                z = torch.rand_like(z) * 2 - 1

            z, x = self.kernel.step_with_preconditioner_inverse(z, update=True)
            x_train_list.append(x)

            target_samples.add(x)
            if return_latent_samples:
                latent_samples.add(z)

            elapsed_time = time.time() - t0
            pbar.set_postfix_str(self.pbar_repr(elapsed_time))
            if time_limit_seconds is not None and elapsed_time > time_limit_seconds:
                break

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
            z, x = self.kernel.step_with_preconditioner_inverse(z)

            target_samples.add(x)
            if return_latent_samples:
                latent_samples.add(z)

            elapsed_time = time.time() - t0
            pbar.set_postfix_str(self.pbar_repr(elapsed_time))
            if time_limit_seconds is not None and elapsed_time > time_limit_seconds:
                break

        if return_latent_samples:
            return target_samples, latent_samples
        return target_samples