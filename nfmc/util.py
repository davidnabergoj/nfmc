def metropolis_acceptance_log_probability(
        target_current_log_prob,
        target_prime_log_prob,
        proposal_current_log_prob,
        proposal_prime_log_prob
):
    # alpha = min(1, p(x_prime)/p(x_curr)*g(x_curr|x_prime)/g(x_prime|x_curr))
    # p = target
    # g(x_curr|x_prime) = proposal_current_log_prob
    # g(x_prime|x_curr) = proposal_prime_log_prob
    return target_prime_log_prob - target_current_log_prob + proposal_current_log_prob - proposal_prime_log_prob
