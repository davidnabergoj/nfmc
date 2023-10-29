import torch


def compute_empirical_moment(samples, k: int):
    """
    Compute empirical moment for each step of a chain.
    :param samples:
    :param k:
    :return:
    """
    # samples.shape = (n_steps, n_dim)
    # output.shape = (n_steps, n_dim)
    n_steps, n_dim = samples.shape
    return torch.cumsum(samples ** k, dim=0) / torch.arange(1, n_steps + 1)[:, None]


def compute_empirical_mean(samples):
    """
    Compute empirical mean for each step of a chain.
    :param samples:
    :return:
    """
    # samples.shape = (n_steps, n_dim)
    # output.shape = (n_steps, n_dim)
    return compute_empirical_moment(samples, 1)


def compute_empirical_2nd_moment(samples):
    """
    Compute empirical 2nd moment for each step of a chain.
    :param samples:
    :return:
    """
    # samples.shape = (n_steps, n_dim)
    # output.shape = (n_steps, n_dim)
    return compute_empirical_moment(samples, 2)


def unnormalized_bias(empirical_moment, true_moment, order: int = 2):
    """
    Compute unnormalized bias for each step of a single chain.
    :param empirical_moment:
    :param true_moment:
    :param order:
    :return:
    """
    # samples.shape = (n_steps, n_dim)
    # true_moment.shape = (n_dim,)
    # output.shape = (n_steps, n_dim)
    # order = 2: squared bias
    # order = 1: squared bias, but numerator is the absolute value
    ret, _ = ((empirical_moment - true_moment[None]) ** order).abs().max(dim=-1)
    return ret


def normalized_bias(empirical_moment, true_moment, true_variance, order: int = 2):
    """
    Compute normalized bias for each step of a single chain.
    :param empirical_moment:
    :param true_moment:
    :param order:
    :return:
    """
    # order = 2: squared bias
    # order = 1: squared bias, but numerator is the absolute value
    # empirical_moment.shape = (n_steps, n_dim)
    # true_moment.shape = (n_dim,)
    ret, _ = (((empirical_moment - true_moment[None]) ** order).abs() / true_variance[None]).max(dim=-1)
    return ret


def absolute_error_mean(samples: torch.Tensor, true_mean: torch.Tensor):
    """
    Compute normalized bias for each step of many chains.
    """
    # samples.shape = (n_steps, n_chains, n_dim)
    # true_mean.shape = (n_dim,)
    # Computes max_{dim} (empirical_mean - true_mean).abs()
    return torch.vmap(
        lambda chain: unnormalized_bias(compute_empirical_mean(chain), true_mean, order=1),
        in_dims=1
    )(samples).mean(dim=0)


def normalized_absolute_error_mean(samples: torch.Tensor, true_mean: torch.Tensor, true_variance: torch.Tensor):
    # samples.shape = (n_steps, n_chains, n_dim)
    # true_mean.shape = (n_dim,)
    # true_variance.shape = (n_dim,)
    # Computes max_{dim} (empirical_mean - true_mean).abs() / true_variance
    # The idea is that some targets have dimensions with different magnitudes. Dividing by the true variance will bring
    # them to the same scale.
    return torch.vmap(
        lambda chain: normalized_bias(compute_empirical_mean(chain), true_mean, true_variance, order=1),
        in_dims=1
    )(samples).mean(dim=0)


def squared_bias_mean(samples: torch.Tensor, true_mean: torch.Tensor, true_variance: torch.Tensor):
    # samples.shape = (n_steps, n_chains, n_dim)
    # true_mean.shape = (n_dim,)
    # Computes max_{dim} (empirical_mean - true_mean) ** 2
    # return normalized_bias(compute_empirical_mean(samples), true_mean, true_variance, order=2)
    return torch.vmap(
        lambda chain: normalized_bias(compute_empirical_mean(chain), true_mean, true_variance, order=2),
        in_dims=1
    )(samples).mean(dim=0)


def absolute_error_2nd_moment(samples: torch.Tensor, true_2nd_moment: torch.Tensor):
    # return unnormalized_bias(compute_empirical_2nd_moment(samples), true_2nd_moment, order=1)
    return torch.vmap(
        lambda chain: unnormalized_bias(compute_empirical_2nd_moment(chain), true_2nd_moment, order=1),
        in_dims=1
    )(samples).mean(dim=0)


def normalized_absolute_error_2nd_moment(samples: torch.Tensor, true_2nd_moment: torch.Tensor,
                                         true_variance: torch.Tensor):
    # return normalized_bias(compute_empirical_2nd_moment(samples), true_2nd_moment, true_variance, order=1)
    return torch.vmap(
        lambda chain: normalized_bias(compute_empirical_2nd_moment(chain), true_2nd_moment, true_variance, order=1),
        in_dims=1
    )(samples).mean(dim=0)


def squared_bias_2nd_moment(samples: torch.Tensor, true_2nd_moment: torch.Tensor, true_variance: torch.Tensor):
    # return normalized_bias(compute_empirical_2nd_moment(samples), true_2nd_moment, true_variance, order=2)
    return torch.vmap(
        lambda chain: normalized_bias(compute_empirical_2nd_moment(chain), true_2nd_moment, true_variance, order=2),
        in_dims=1
    )(samples).mean(dim=0)


def steps_to_low_squared_bias_2nd_moment(samples: torch.Tensor,
                                         true_2nd_moment: torch.Tensor,
                                         true_variance: torch.Tensor):
    # How many steps do we need before reaching squared bias of 0.01
    b2 = squared_bias_2nd_moment(samples, true_2nd_moment, true_variance)
    if torch.all(b2 >= 0.01):
        # Never reached 0.01
        steps_until_goal = torch.inf
    else:
        steps_until_goal = torch.where(b2 < 0.01)[0][0]
    return steps_until_goal
