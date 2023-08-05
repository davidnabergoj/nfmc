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
