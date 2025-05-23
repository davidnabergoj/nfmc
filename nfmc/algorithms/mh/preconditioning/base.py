from copy import deepcopy
import time
from typing import List, Tuple, Union
import torch
from tqdm import tqdm
from nfmc.algorithms.mh.base import MHKernel
from nfmc.algorithms.sampling.base.sampler import MHSampler
from nfmc.algorithms.util.samples import Samples


class Preconditioner:
    """
    MCMC preconditioning class.
    """

    def __init__(self):
        pass

    def inverse_transform(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the inverse of z under this preconditioner.

        :param torch.Tensor z: latent tensor with shape `(*batch_shape, *event_shape)`.
        :return: tuple where the first element is the transformed latent tensor with shape 
         `(*batch_shape, *event_shape)` and the second element is the log of the absolute value of the Jacobian 
         determinant of this inverse transformation with respect to the latent tensor with shape `batch_shape`.
        """
        raise NotImplementedError

    def fit(self, x: torch.Tensor):
        pass


class PreconditionedMHKernel(MHKernel):
    """
    Preconditioned Metropolis-Hastings kernel class.

    All kernel transitions are performed according to a preconditioner-adjusted target density.
    """

    def __init__(self,
                 base_kernel: MHKernel,
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

        # Override the target log probability density
        self.base_kernel.neg_log_prob_target = self.neg_log_prob_adjusted_target

    def neg_log_prob_adjusted_target(self, z: torch.Tensor) -> torch.Tensor:
        """
        Returns the negative log probability density of the preconditioner-adjusted target distribution.

        :param torch.Tensor z: latent tensor with shape `(*batch_shape, *event_shape)`.
        :return: negative log probability tensor with shape `batch_shape`
        """
        x, log_det_inverse = self.preconditioner.inverse_transform(z)
        return self.neg_log_prob_target(x) - log_det_inverse

    def step(self,
             z: torch.Tensor,
             update_kernel: bool = False):
        """
        Performs one kernel transition.

        :param torch.Tensor z: current latent state tensor with shape `(*batch_shape, *event_shape)`.
        :param bool update_kernel: if True, update kernel parameters.
        :return: new latent state tensor with shape `(*batch_shape, *event_shape)`.
        """
        return self.base_kernel.step(z, update=update_kernel)


class PreconditionedMHSampler(MHSampler):
    """
    Sampler class for Metropolis-Hastings algorithms with preconditioning.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 kernel: PreconditionedMHKernel,
                 **kwargs):
        self.event_shape = event_shape
        self.kernel = kernel

    @property
    def name(self) -> str:
        return "Generic preconditioned MH sampler"

    def prepare_training_data(self,
                              train_data_list: List[torch.Tensor],
                              max_training_samples: int):
        data_shape = train_data_list[0].shape
        n_batch_dims = len(data_shape) - len(self.kernel.event_shape)
        batch_dims = list(range(n_batch_dims))
        z_train = torch.cat(train_data_list, dim=batch_dims)
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
               data_transform: callable = None) -> Samples:
        """
        Optimizes kernel parameters.

        The kernel is updated every step unless it internally overrides this.
        The preconditioner is updated every K steps where K is equal to preconditioner_update_interval.

        :param torch.Tensor z0: initial latent state with shape `(*batch_shape, *event_shape)`.
        :param int n_steps: number of MCMC steps to perform.
        :param int preconditioner_update_interval: update the preconditioner after this number of MCMC steps.
        :param bool show_progress: if True, display a progress bar.
        :param float time_limit_seconds: maximum sampling time. Sampling stops if this time is exceeded.
        :param int max_samples: maximum number of samples to store.
        :param int max_samples: maximum number of training samples to train the preconditioner.
        :param callable data_transform: function that transforms each generated sample. Receives as input a tensor with
         shape `(*batch_shape, *event_shape)` and outputs a tensor with shape `(*batch_shape, *event_shape)`.
        """
        target_samples = Samples(
            event_shape=self.kernel.event_shape,
            max_samples=max_samples,
            data_transform=data_transform
        )
        self.kernel.reset_statistics()
        z = deepcopy(z0.detach())

        # Holds data for preconditioner training/fitting
        z_train_list = []

        t0 = time.time()
        for step in (pbar := tqdm(range(n_steps),
                                  desc=f'{self.kernel.name} sampling',
                                  disable=not show_progress)):

            if step % preconditioner_update_interval == 0 and step > 0:
                # Update the preconditioner first so drawn sample can contribute toward next preconditioner fit.
                z_train = self.prepare_training_data(
                    z_train_list,
                    max_training_samples
                )
                self.kernel.preconditioner.fit(x=z_train)
                z_train_list = []

            z = self.kernel.step(z, update_kernel=True)
            target_samples.add(z)
            z_train_list.append(z)

            elapsed_time = time.time() - t0
            pbar.set_postfix_str(
                f'acc-rate: {self.kernel.acceptance_rate} | '
                f'calls/s: {self.calls_per_second(elapsed_time)} | '
                f'grads/s: {self.grads_per_second(elapsed_time)} | '
            )
            if time_limit_seconds is not None and elapsed_time > time_limit_seconds:
                break

        return target_samples

    def sample(self,
               x0: torch.Tensor,
               n_steps: int,
               show_progress: bool = True,
               time_limit_seconds: Union[float, int] = None) -> Samples:
        """
        Samples with a fixed kernel.
        """
        raise NotImplementedError
