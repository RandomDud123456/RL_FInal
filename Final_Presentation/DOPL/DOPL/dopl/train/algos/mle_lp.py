import numpy as np
from scipy.optimize import minimize
import time
from tqdm import tqdm
import numpy as np
from dopl.utils import wandb_log_latest
from dopl.utils.lp_solver import compute_ELP_pyomo
from dopl.registry import register_training_function
from dopl.train.helpers import apply_index_policy, compute_F_true
from scipy.stats import kendalltau


@register_training_function("mle_lp")
def train(env, cfg):
    start_time = time.time()
    K = cfg["K"]
    eps = cfg["eps"]

    P_true = np.array(env.P_list)
    R_true = np.array(env.R_list)[:, :, 0]

    num_arms = len(env.P_list)
    num_states = env.P_list[0].shape[0]
    num_actions = env.R_list[0].shape[1]

    P_hat = np.ones((num_arms, num_states, num_states, num_actions)) / num_states

    Z_sa = np.zeros((num_arms, num_states, num_actions))
    Z_sas = np.zeros((num_arms, num_states, num_states, num_actions))
    R_est = np.ones((num_arms, num_states)) * 0.5
    delta = np.ones((num_arms, num_states, num_actions))

    metrics = {
        "reward": [],
        "index_error": [],
        "P_error": [],
        "R_error": [],
        "run_time": 0,
    }

    W_sas = None
    # battle_data = []
    for k in tqdm(range(K)):

        # LP solve using R_est to get index policy
        W_sas, _ = compute_ELP_pyomo(
            delta,
            P_hat,
            env.arm_constraint,
            num_states,
            num_actions,
            R_est,
            num_arms,
        )

        W_sa = np.sum(W_sas, axis=2)
        W0 = W_sa[:, :, 0]
        W1 = W_sa[:, :, 1]
        denom = W0 + W1
        index_matrix_pre_nan = np.divide(
            W1, denom, out=np.zeros_like(W1), where=denom != 0
        )
        index_matrix = np.nan_to_num(index_matrix_pre_nan, nan=0.0)

        # Keep track of errors
        metrics["index_error"].append(
            kendalltau(index_matrix.ravel(), env.opt_index.ravel())[0]
        )  # Kendall tau coeff : -1 means opposite, 0 means no correlation, 1 means same order
        metrics["P_error"].append(np.linalg.norm(P_hat - P_true))
        metrics["R_error"].append(np.linalg.norm(R_est - R_true))

        # Start rollout using index policy
        battle_data = []
        s_list = env.reset()
        reward_episode = 0
        for t in range(env.H):
            action = apply_index_policy(s_list, index_matrix, env.arm_constraint)
            s_dash_list, reward, _, _, info = env.step(action)
            reward_episode += reward
            # Update P estimate
            for arm_id, s, a, s_dash in zip(
                range(num_arms), s_list, action, s_dash_list
            ):
                Z_sa[arm_id, s, a] += 1
                Z_sas[arm_id, s, s_dash, a] += 1
                delta[arm_id, s, a] = np.sqrt(
                    np.log(
                        4 * num_states * num_actions * num_arms * (k + 1) * env.H / eps
                    )
                    / (2 * Z_sa[arm_id, s, a])
                )
                P_hat[arm_id, s, s_dash, a] = Z_sas[arm_id, s, s_dash, a] / np.maximum(
                    1, Z_sa[arm_id, s, a]
                )
            # Update battle data to fit Bradley-Terry model using MLE
            for record in info["duelling_results"]:
                winner, loser = record
                battle_data.append([winner, s_list[winner], loser, s_list[loser]])
            s_list = s_dash_list
        R_est = mle_bradley_terry(np.array(battle_data), R_est)
        metrics["reward"].append(reward_episode)

        wandb_log_latest(metrics)

    end_time = time.time()
    metrics["run_time"] = end_time - start_time
    return metrics


def neg_log_likelihood(R_flat, comparisons, num_arms, num_states):
    """Calculate the negative log-likelihood for the Bradley-Terry model using vectorized operations."""
    R = R_flat.reshape((num_arms, num_states))
    exp_R = np.exp(R)

    # Extract the indices for winners and losers
    win_indices = (comparisons[:, 0], comparisons[:, 1])
    lose_indices = (comparisons[:, 2], comparisons[:, 3])

    # Calculate the probabilities in a vectorized way
    win_probs = exp_R[win_indices]
    lose_probs = exp_R[lose_indices]
    total_probs = win_probs + lose_probs
    probabilities = win_probs / total_probs

    # Compute log-likelihood
    log_likelihood = np.sum(
        np.log(probabilities + 1e-20)
    )  # Small constant to avoid log(0)
    return -log_likelihood


def mle_bradley_terry(comparisons, R_est):
    """Estimate parameters using scipy.optimize.minimize with vectorized likelihood computation."""
    num_arms, num_states = R_est.shape
    initial_guess = R_est.flatten()  # Flatten R for optimization
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
        return result.x.reshape((num_arms, num_states))  # Reshape R back to matrix form
    else:
        raise RuntimeError("Optimization did not converge: " + result.message)


if __name__ == "__main__":
    # Example usage
    comparisons = np.array(
        [
            (0, 1, 1, 0),  # (winning_arm, winning_state, losing_arm, losing_state)
            (1, 0, 0, 1),
            (0, 1, 1, 1),
        ]
    )

    num_arms = 2
    num_states = 2
    R_est = np.ones((num_arms, num_states)) * 0.5  # Initial guess for R
    optimized_R = mle_bradley_terry(comparisons, R_est)
    print("Optimized R parameters:")
    print(optimized_R)
