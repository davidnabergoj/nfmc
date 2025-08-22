from copy import deepcopy
import time
from typing import Union
import torch
from torchflows import Flow
from tqdm import tqdm
from nfmc.algorithms.kernel import MixingKernel
from nfmc.algorithms.mh.base import MHKernel
from nfmc.algorithms.mh.local.mala import MALAKernel
from nfmc.algorithms.mh.local.hmc import HMCKernel
from nfmc.algorithms.mh.local.rwmh import RWMHKernel
from nfmc.algorithms.preconditioning.preconditioners import DiagonalLinearPreconditioner, NormalizingFlowPreconditioner
from nfmc.algorithms.preconditioning.samplers.base import PreconditionedMCMCSampler
from nfmc.algorithms.util.samples import Samples


class BinaryMixedPreconditionedMCMCSampler(PreconditionedMCMCSampler):
    """
    Base class for mixed preconditioned MCMC samplers containing two kernels.
    Intended use: the first kernel is simple and fast, the second kernel is more complex and slow.
    """

    def __init__(self, kernel1: MHKernel, kernel2: MHKernel, warmup_prob_schedule: callable = None):
        """
        BinaryMixedPreconditionedMCMCSampler constructor.

        :param MHKernel kernel1: first kernel object.
        :param MHKernel kernel2: second kernel object.
        :param callable warmup_prob_schedule: function that computes the first kernel's selection probability based on the current warmup iteration index.
         If None, the selection probability is unchanged from the initial value.
        """
        mixing_kernel = MixingKernel([kernel1, kernel2])
        super().__init__(mixing_kernel)
        self.warmup_prob_schedule = warmup_prob_schedule
        if self.warmup_prob_schedule is not None:
            p1 = self.warmup_prob_schedule(0)
            p2 = 1.0 - p1
            self.kernel.set_selection_probabilities(
                [p1, p2]
            )

    def warmup(self,
               x0: torch.Tensor,
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

        The mixing kernel's selection probabilities are updated after every preconditioner update.

        :param torch.Tensor x0: initial target-space state with shape `(*batch_shape, *event_shape)`.
        :param int n_steps: number of MCMC steps to perform.
        :param int preconditioner_update_interval: update the preconditioner after this number of MCMC steps.
        :param bool show_progress: if True, display a progress bar.
        :param float time_limit_seconds: maximum sampling time. Sampling stops if this time is exceeded.
        :param int max_samples: maximum number of samples to store.
        :param int max_training_samples: maximum number of training samples to train the preconditioner.
        :param callable data_transform: function that transforms each generated sample. Receives as input a tensor with
         shape `(*batch_shape, *event_shape)` and outputs a tensor with shape `(*batch_shape, *event_shape)`.
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

        _adj_max = max_training_samples
        if max_training_samples is not None:
            _adj_max /= len(x0)  # divide by number of chains
        training_samples = Samples(
            event_shape=self.kernel.event_shape,
            max_samples=_adj_max,
        )

        self.kernel.reset_statistics()
        x = deepcopy(x0.detach())

        t0 = time.time()
        for step in (pbar := tqdm(range(n_steps),
                                  desc=f'Warmup',
                                  disable=not show_progress)):
            if step % preconditioner_update_interval == 0 and 0 < step <= n_steps - preconditioner_update_interval:
                # Update the preconditioner first so drawn sample can contribute toward next preconditioner fit.
                x_train = training_samples.as_tensor().view(-1, *self.kernel.event_shape)

                self.kernel.fit_preconditioner(x_train, **kwargs)
                training_samples = Samples(
                    event_shape=self.kernel.event_shape,
                    max_samples=_adj_max,
                )

                # Reset kernel
                self.kernel.reset_parameters()

                # Set selection probabilities for the mixing kernel
                if self.warmup_prob_schedule is not None:
                    p1 = self.warmup_prob_schedule(step)
                    p2 = 1.0 - p1
                    self.kernel.set_selection_probabilities([p1, p2])

            # Step. Update if at least K // 2 steps from the next preconditioner update.
            do_update = (
                step % preconditioner_update_interval < (
                    preconditioner_update_interval // 2)  # not too close
                # final stage: always update
                or step >= (n_steps - preconditioner_update_interval)
            )

            z = self.kernel._preconditioner.forward_transform(x)[0]
            _, x = self.kernel.step_with_preconditioner_inverse(
                z,
                update=do_update
            )
            if do_update:
                training_samples.add(x)

            target_samples.add(x)
            if return_latent_samples:
                latent_samples.add(z)

            elapsed_time = time.time() - t0
            pbar.set_postfix_str(self.kernel.pbar_repr(elapsed_time))
            if time_limit_seconds is not None and elapsed_time > time_limit_seconds:
                break

        

        if return_latent_samples:
            return target_samples, latent_samples
        return target_samples


class MixingNeuTraRWMH(BinaryMixedPreconditionedMCMCSampler):
    """
    RWMH sampler that combines normalizing flow and diagonal preconditioning.
    """

    def __init__(self,
                 flow: Flow,
                 neg_log_prob_target: callable,
                 warmup_prob_schedule: callable = None,
                 **kwargs):
        """
        NeuTraRWMH constructor.

        :param Flow flow: normalizing flow object.
        :param neg_log_prob_target: negative log probability density callable.
        """

        super().__init__(
            RWMHKernel(
                flow.event_shape,
                neg_log_prob_target,
                preconditioner=DiagonalLinearPreconditioner(flow.event_shape),
                **kwargs
            ),
            RWMHKernel(
                flow.event_shape,
                neg_log_prob_target,
                preconditioner=NormalizingFlowPreconditioner(flow),
                **kwargs
            ),
            warmup_prob_schedule=warmup_prob_schedule
        )


class MixingNeuTraMALA(BinaryMixedPreconditionedMCMCSampler):
    """
    MALA sampler that combines normalizing flow and diagonal preconditioning.
    """

    def __init__(self,
                 flow: Flow,
                 neg_log_prob_target: callable,
                 warmup_prob_schedule: callable = None,
                 **kwargs):
        """
        NeuTraMALA constructor.

        :param Flow flow: normalizing flow object.
        :param neg_log_prob_target: negative log probability density callable.
        """
        super().__init__(
            MALAKernel(
                flow.event_shape,
                neg_log_prob_target,
                preconditioner=DiagonalLinearPreconditioner(flow.event_shape),
                **kwargs
            ),
            MALAKernel(
                flow.event_shape,
                neg_log_prob_target,
                preconditioner=NormalizingFlowPreconditioner(flow),
                **kwargs
            ),
            warmup_prob_schedule=warmup_prob_schedule
        )


class MixingNeuTraHMC(BinaryMixedPreconditionedMCMCSampler):
    """
    HMC sampler that combines normalizing flow and diagonal preconditioning.
    """

    def __init__(self,
                 flow: Flow,
                 neg_log_prob_target: callable,
                 warmup_prob_schedule: callable = None,
                 **kwargs):
        """
        NeuTraHMC constructor.

        :param Flow flow: normalizing flow object.
        :param neg_log_prob_target: negative log probability density callable.
        """
        super().__init__(
            HMCKernel(
                flow.event_shape,
                neg_log_prob_target,
                preconditioner=DiagonalLinearPreconditioner(flow.event_shape),
                **kwargs
            ),
            HMCKernel(
                flow.event_shape,
                neg_log_prob_target,
                preconditioner=NormalizingFlowPreconditioner(flow),
                **kwargs
            ),
            warmup_prob_schedule=warmup_prob_schedule
        )
