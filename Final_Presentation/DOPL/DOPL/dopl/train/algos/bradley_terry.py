import numpy as np
from scipy.optimize import minimize


def neg_log_likelihood(R_flat, comparisons, num_arms, num_states):
    """Calculate the negative log-likelihood for the Bradley-Terry model."""
    R = R_flat.reshape((num_arms, num_states))
    exp_R = np.exp(R)

    win_indices = (comparisons[:, 0], comparisons[:, 1])
    lose_indices = (comparisons[:, 2], comparisons[:, 3])

    win_probs = exp_R[win_indices]
    lose_probs = exp_R[lose_indices]
    total_probs = win_probs + lose_probs
    probabilities = win_probs / total_probs

    log_likelihood = np.sum(np.log(probabilities + 1e-20))
    return -log_likelihood


def mle_bradley_terry(comparisons, R_est):
    """Estimate Bradley-Terry parameters using scipy.optimize.minimize."""
    num_arms, num_states = R_est.shape
    initial_guess = R_est.flatten()
    bounds = [(0, 1) for _ in range(num_arms * num_states)]

    result = minimize(
        neg_log_likelihood,
        initial_guess,
        args=(comparisons, num_arms, num_states),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 10000000, "gtol": 1e-3},
    )

    if result.success:
        return result.x.reshape((num_arms, num_states))
    raise RuntimeError("Optimization did not converge: " + result.message)
