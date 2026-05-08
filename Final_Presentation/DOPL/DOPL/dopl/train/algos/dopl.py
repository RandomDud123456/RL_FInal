import numpy as np
from dopl.train.helpers import enrich_F, pick_best_ref
import numpy as np
from tqdm import tqdm
from dopl.utils import wandb_log_latest
from dopl.utils.lp_solver import compute_ELP_pyomo
from dopl.train.helpers import apply_index_policy, compute_F_true
from dopl.registry import register_training_function
import time
from scipy.stats import kendalltau


@register_training_function("dopl")
def train(env, cfg):
    start_time = time.time()
    K = cfg["K"]
    eps = cfg["eps"]

    P_true = np.array(env.P_list)
    F_true = compute_F_true(env)

    num_arms = len(env.P_list)
    num_states = env.P_list[0].shape[0]
    num_actions = env.R_list[0].shape[1]

    # battle_data = []
    Wins = np.zeros((num_arms, num_states, num_arms, num_states))
    P_hat = np.ones((num_arms, num_states, num_states, num_actions)) / num_states

    Z_sa = np.zeros((num_arms, num_states, num_actions))
    Z_sas = np.zeros((num_arms, num_states, num_states, num_actions))
    F_tilde = np.ones((num_arms, num_states, num_arms, num_states)) * (
        np.e / (np.e + 1) if cfg.reward_normalized else 1 - 1e-6
    )
    F_hat = np.ones((num_arms, num_states, num_arms, num_states)) * 0.5
    # R_est = np.ones((num_arms, num_states)) * 0.5
    delta = np.ones((num_arms, num_states, num_actions))
    conf = np.ones((num_arms, num_states, num_arms, num_states)) * np.sqrt(
        np.log(4 * num_states * num_actions * num_arms * K * env.H / eps) / 2
    )

    for arm in range(num_arms):
        for state in range(num_states):
            F_hat[arm, state, arm, state] = 0.5
            conf[arm, state, arm, state] = 0.0
            F_tilde[arm, state, arm, state] = 0.5

    metrics = {
        "reward": [],
        "index_error": [],
        "F_error": [],
        "P_error": [],
        "R_error": [],
        "run_time": 0,
    }

    W_sas = None
    for k in tqdm(range(K)):

        # Enrich F estimate
        ref_arm, ref_state = pick_best_ref(Wins)
        F_tilde, F_hat = enrich_F(F_tilde, F_hat, (ref_arm, ref_state), conf)
        if cfg.reward_normalized:
            F_tilde = np.clip(F_tilde, 1 / (np.e + 1), np.e / (np.e + 1))
            F_hat = np.clip(F_hat, 1 / (np.e + 1), np.e / (np.e + 1))
        else:
            F_tilde = np.clip(F_tilde, 1e-6, 1 - 1e-6)
            F_hat = np.clip(F_hat, 1e-6, 1 - 1e-6)

        # Use F estimate to compute index (policy)
        Q_n_s = np.log(
            F_tilde[:, :, ref_arm, ref_state] / (1 - F_tilde[:, :, ref_arm, ref_state])
        )

        W_sas, _ = compute_ELP_pyomo(
            delta,
            P_hat,
            env.arm_constraint,
            num_states,
            num_actions,
            Q_n_s,
            num_arms,
        )

        W_sa = np.sum(W_sas, axis=2)
        W0 = W_sa[:, :, 0]
        W1 = W_sa[:,:,1]
        denom = W0 + W1
        index_matrix_pre_nan = np.divide(W1, denom, out=np.zeros_like(W1), where=denom!=0)
        index_matrix = np.nan_to_num(index_matrix_pre_nan, nan=0.0)

        # Keep track of errors
        Q_true = np.log(
            F_true[:, :, ref_arm, ref_state] / (1 - F_true[:, :, ref_arm, ref_state])
        )
        metrics["R_error"].append(np.linalg.norm(Q_n_s - Q_true))
        metrics["index_error"].append(kendalltau(index_matrix.ravel(), env.opt_index.ravel())[0]) # Kendall tau coeff : -1 means opposite, 0 means no correlation, 1 means same order
        metrics["F_error"].append(np.linalg.norm(F_hat - F_true))
        metrics["P_error"].append(np.linalg.norm(P_hat - P_true))

        # Start rollout using index policy
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
            # Update F_estimate
            for record in info["duelling_results"]:
                winner, loser = record
                Wins[winner, s_list[winner], loser, s_list[loser]] += 1
                battle_count = (
                    Wins[winner, s_list[winner], loser, s_list[loser]]
                    + Wins[loser, s_list[loser], winner, s_list[winner]]
                )
                F_hat[winner, s_list[winner], loser, s_list[loser]] = (
                    Wins[winner, s_list[winner], loser, s_list[loser]] / battle_count
                )
                F_hat[loser, s_list[loser], winner, s_list[winner]] = (
                    1 - F_hat[winner, s_list[winner], loser, s_list[loser]]
                )
                conf[winner, s_list[winner], loser, s_list[loser]] = conf[
                    loser, s_list[loser], winner, s_list[winner]
                ] = np.sqrt(
                    np.log(4 * num_states * num_arms * (k + 1) * env.H / eps)
                    / (2 * battle_count)
                )

                F_tilde[winner, s_list[winner], loser, s_list[loser]] = (
                    F_hat[winner, s_list[winner], loser, s_list[loser]]
                    + conf[winner, s_list[winner], loser, s_list[loser]]
                )
                F_tilde[loser, s_list[loser], winner, s_list[winner]] = (
                    F_hat[loser, s_list[loser], winner, s_list[winner]]
                    + conf[loser, s_list[loser], winner, s_list[winner]]
                )
                if cfg.reward_normalized:
                    F_tilde = np.clip(
                        F_tilde, 1 / (np.e + 1), np.e / (np.e + 1)
                    )  # Since reward bounded between 0 and 1
                    F_hat = np.clip(F_hat, 1 / (np.e + 1), np.e / (np.e + 1))
                else:
                    F_tilde = np.clip(
                        F_tilde, 1e-6, 1 - 1e-6
                    )  # For numerical stability
                    F_hat = np.clip(F_hat, 1e-6, 1 - 1e-6)
            s_list = s_dash_list

        metrics["reward"].append(reward_episode)

        wandb_log_latest(metrics)

    end_time = time.time()
    metrics["run_time"] = end_time - start_time
    return metrics
