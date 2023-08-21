import torch


def empirical_mean(samples):
    # samples.shape = (n_steps, n_chains, n_dim)
    n_steps, n_chains, n_dim = samples.shape
    tmp = torch.sum(samples, dim=1)
    return torch.cumsum(tmp, dim=0) / (n_chains * torch.arange(n_steps))[:, None]


def empirical_2nd_moment(samples):
    # samples.shape = (n_steps, n_chains, n_dim)
    n_steps, n_chains, n_dim = samples.shape
    tmp = torch.sum(samples ** 2, dim=1)
    return torch.cumsum(tmp, dim=0) / (n_chains * torch.arange(n_steps))[:, None]


def unnormalized_bias(empirical_moment, true_moment, order: int = 2):
    # order = 2: squared bias
    # order = 1: squared bias, but numerator is the absolute value
    # empirical_moment.shape = (n_steps, n_dim)
    # true_moment.shape = (n_dim,)
    ret, _ = ((empirical_moment - true_moment[None]) ** order).abs().max(dim=-1)
    return ret


def normalized_bias(empirical_moment, true_moment, true_variance, order: int = 2):
    # order = 2: squared bias
    # order = 1: squared bias, but numerator is the absolute value
    # empirical_moment.shape = (n_steps, n_dim)
    # true_moment.shape = (n_dim,)
    ret, _ = (((empirical_moment - true_moment[None]) ** order).abs() / true_variance[None]).max(dim=-1)
    return ret


def absolute_error_mean(samples: torch.Tensor, true_mean: torch.Tensor):
    # samples.shape = (n_steps, n_chains, n_dim)
    # true_mean.shape = (n_dim,)
    # Computes max_{dim} (empirical_mean - true_mean).abs()
    return unnormalized_bias(empirical_mean(samples), true_mean, order=1)


def normalized_absolute_error_mean(samples: torch.Tensor, true_mean: torch.Tensor, true_variance: torch.Tensor):
    # samples.shape = (n_steps, n_chains, n_dim)
    # true_mean.shape = (n_dim,)
    # true_variance.shape = (n_dim,)
    # Computes max_{dim} (empirical_mean - true_mean).abs() / true_variance
    # The idea is that some targets have dimensions with different magnitudes. Dividing by the true variance will bring
    # them to the same scale.
    return normalized_bias(empirical_mean(samples), true_mean, true_variance, order=1)


def squared_bias_mean(samples: torch.Tensor, true_mean: torch.Tensor, true_variance: torch.Tensor):
    # samples.shape = (n_steps, n_chains, n_dim)
    # true_mean.shape = (n_dim,)
    # Computes max_{dim} (empirical_mean - true_mean) ** 2
    return normalized_bias(empirical_mean(samples), true_mean, true_variance, order=2)


def absolute_error_2nd_moment(samples: torch.Tensor, true_2nd_moment: torch.Tensor):
    return unnormalized_bias(empirical_2nd_moment(samples), true_2nd_moment, order=1)


def normalized_absolute_error_2nd_moment(samples: torch.Tensor, true_2nd_moment: torch.Tensor,
                                         true_variance: torch.Tensor):
    return normalized_bias(empirical_2nd_moment(samples), true_2nd_moment, true_variance, order=1)


def squared_bias_2nd_moment(samples: torch.Tensor, true_2nd_moment: torch.Tensor, true_variance: torch.Tensor):
    return normalized_bias(empirical_2nd_moment(samples), true_2nd_moment, true_variance, order=2)
