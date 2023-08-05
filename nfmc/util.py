import math


def metropolis_acceptance_log_ratio(
        log_prob_curr,
        log_prob_prime,
        log_proposal_curr,
        log_proposal_prime
):
    # alpha = min(1, p(x_prime)/p(x_curr)*g(x_curr|x_prime)/g(x_prime|x_curr))
    # p = target
    # g(x_curr|x_prime) = log_proposal_curr
    # g(x_prime|x_curr) = log_proposal_prime
    return log_prob_prime - log_prob_curr + log_proposal_curr - log_proposal_prime


class DualAveraging:
    def __init__(self, initial_value):
        self.h_sum = 0.
        self.x_bar = initial_value
        self.t = 0
        self.kappa = 0.75
        self.mu = math.log(10)
        self.gamma = 0.05
        self.t0 = 10

    def step(self, h_new):
        self.t += 1
        eta = self.t ** -self.kappa
        self.h_sum += h_new
        x_new = self.mu - math.sqrt(self.t) / self.gamma / (self.t + self.t0) * self.h_sum
        x_new_bar = eta * x_new + (1 - eta) * self.x_bar
        self.x_bar = x_new_bar

    def __call__(self):
        return self.x_bar
