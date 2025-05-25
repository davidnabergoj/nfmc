from copy import deepcopy
import time
from typing import List, Tuple, Union
import torch
import torch.nn as nn
from tqdm import tqdm
from nfmc.algorithms.mh.base import MarkovKernel
from nfmc.algorithms.sampling.base.sampler import MCMCSampler
from nfmc.algorithms.util.samples import Samples


class Preconditioner(nn.Module):
    """
    MCMC preconditioning class.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__()
        self.event_shape = event_shape

    def inverse_transform(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the inverse of z under this preconditioner.

        :param torch.Tensor z: latent tensor with shape `(*batch_shape, *event_shape)`.
        :return: tuple where the first element is the transformed latent tensor with shape 
         `(*batch_shape, *event_shape)` and the second element is the log of the absolute value of the Jacobian 
         determinant of this inverse transformation with respect to the latent tensor with shape `batch_shape`.
        """
        raise NotImplementedError

    def fit(self, z: torch.Tensor, **kwargs):
        """
        Update parameters of this preconditioner.

        :param torch.Tensor z: tensor of samples with shape `(*batch_shape, *event_shape)`.
        :param kwargs:
        """
        raise NotImplementedError


class PreconditionedMarkovKernel(MarkovKernel):
    """
    Preconditioned Markov kernel class.

    All kernel transitions are performed according to a preconditioner-adjusted target density.
    """

    def __init__(self,
                 base_kernel: MarkovKernel,
                 preconditioner: Preconditioner):
        super().__init__(
            event_shape=base_kernel.event_shape,
            neg_log_prob_target=base_kernel.neg_log_prob_target
        )

        self.preconditioner = preconditioner
        self.base_neg_log_prob_target = deepcopy(
            base_kernel.neg_log_prob_target
        )
        self.base_kernel = base_kernel

    @property
    def name(self):
        return f'Preconditioned {self.base_kernel.name}'

    def neg_log_prob_adjusted_target(self, z: torch.Tensor) -> torch.Tensor:
        """
        Returns the negative log probability density of the preconditioner-adjusted target distribution.

        :param torch.Tensor z: latent tensor with shape `(*batch_shape, *event_shape)`.
        :return: negative log probability tensor with shape `batch_shape`
        """
        x, log_det_inverse = self.preconditioner.inverse_transform(z)
        return self.base_neg_log_prob_target(x) - log_det_inverse

    def step(self,
             z: torch.Tensor,
             update: bool = False):
        """
        Performs one kernel transition.

        :param torch.Tensor z: current latent state tensor with shape `(*batch_shape, *event_shape)`.
        :param bool update: if True, update base kernel parameters.
        :return: new latent state tensor with shape `(*batch_shape, *event_shape)`.
        """
        # Ensure the correct target distribution is used
        self.base_kernel.set_target(self.neg_log_prob_adjusted_target)
        return self.base_kernel.step(z, update=update)


class PreconditionedMCMCSampler(MCMCSampler):
    """
    Sampler class for MCMC algorithms with preconditioning.
    """

    def __init__(self,
                 kernel: PreconditionedMarkovKernel,
                 **kwargs):
        super().__init__(kernel)

    @property
    def name(self) -> str:
        return "Generic preconditioned MCMC sampler"

    def prepare_training_data(self,
                              train_data_list: List[torch.Tensor],
                              max_training_samples: int):
        # Flatten training data elements
        train_data_list = [z.view(-1, *self.kernel.event_shape) for z in train_data_list]
        z_train = torch.concat(train_data_list, dim=0)
        z_train = z_train[torch.randperm(len(z_train))]
        if max_training_samples is not None:
            z_train = z_train[:max_training_samples]
        return z_train

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
        The preconditioner is updated every K steps where K is equal to preconditioner_update_interval.
        The preconditioner is not updated within the final K steps so that the rest of the kernel can be stably tuned.

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
        if data_transform is None:
            data_transform = lambda v: v

        target_samples = Samples(
            event_shape=self.kernel.event_shape,
            max_samples=max_samples,
            data_transform=lambda z: data_transform(
                self.kernel.preconditioner.inverse_transform(z)[0]
            )
        )
        latent_samples = Samples(
            event_shape=self.kernel.event_shape,
            max_samples=max_samples,
        )

        self.kernel.reset_statistics()
        z = deepcopy(z0.detach())

        # Holds data for preconditioner training/fitting
        z_train_list = []

        t0 = time.time()
        for step in (pbar := tqdm(range(n_steps),
                                  desc=f'{self.kernel.name} sampling',
                                  disable=not show_progress)):

            if step % preconditioner_update_interval == 0 and 0 < step < n_steps - preconditioner_update_interval:
                # Update the preconditioner first so drawn sample can contribute toward next preconditioner fit.
                z_train = self.prepare_training_data(
                    z_train_list,
                    max_training_samples
                )
                self.kernel.preconditioner.fit(z=z_train, **kwargs)
                z_train_list = []

            z = self.kernel.step(z, update=True)
            target_samples.add(z)
            if return_latent_samples:
                latent_samples.add(z)
            z_train_list.append(z)

            elapsed_time = time.time() - t0
            pbar.set_postfix_str(
                f'calls/s: {self.calls_per_second(elapsed_time)} | '
                f'grads/s: {self.grads_per_second(elapsed_time)} | '
            )
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
        The preconditioner is updated every K steps where K is equal to preconditioner_update_interval.

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
        if data_transform is None:
            data_transform = lambda v: v

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
                               desc=f'{self.kernel.name} sampling',
                               disable=not show_progress)):
            z = self.kernel.step(z)
            target_samples.add(z)
            if return_latent_samples:
                latent_samples.add(z)

            elapsed_time = time.time() - t0
            pbar.set_postfix_str(
                f'calls/s: {self.calls_per_second(elapsed_time)} | '
                f'grads/s: {self.grads_per_second(elapsed_time)} | '
            )
            if time_limit_seconds is not None and elapsed_time > time_limit_seconds:
                break

        if return_latent_samples:
            return target_samples, latent_samples
        return target_samples
