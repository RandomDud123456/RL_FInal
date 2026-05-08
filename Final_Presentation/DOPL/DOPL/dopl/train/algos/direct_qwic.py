from tqdm import tqdm
import numpy as np
from dopl.utils import wandb_log_latest
from dopl.registry import register_training_function
import torch
from dopl.train.algos.bradley_terry import mle_bradley_terry
from dopl.train.helpers import apply_index_policy, compute_F_true, pick_best_ref, enrich_F
import time
from scipy.stats import kendalltau

@register_training_function("direct_qwic")
def train(env, cfg):
    start_time = time.time()
    K = cfg["K"]
    eps = cfg["eps"]
    R_true = np.array(env.R_list)[:, :, 0]
    F_true = compute_F_true(env)

    num_arms = len(env.P_list)
    num_states = env.P_list[0].shape[0]
    num_actions = env.R_list[0].shape[1]
    lambda_candidates = list(np.linspace(-1, 1, num_arms * num_states))

    R_est = np.ones((num_arms, num_states)) * 0.5
    Q = np.random.rand(
        num_arms,
        len(lambda_candidates),
        num_states,
        num_actions,
    )
    W = np.random.choice(lambda_candidates, size=(num_arms, num_states))
    Wins = np.zeros((num_arms, num_states, num_arms, num_states))
    Z_sa = np.zeros((num_arms, num_states, num_actions))
    F_tilde = np.ones((num_arms, num_states, num_arms, num_states)) * (
        np.e / (np.e + 1) if cfg.reward_normalized else 1 - 1e-6
    )
    F_hat = np.ones((num_arms, num_states, num_arms, num_states)) * 0.5
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
        "R_error": [],
        "F_error": [],
        "run_time": 0,
    }
    global_step = 0
    for k in tqdm(range(K)):
        # Start rollout using Q values as policy
        battle_data = []
        traj_states = []
        traj_actions = []
        traj_next_states = []
        s_list = env.reset()
        reward_episode = 0
        for t in range(env.H):
            global_step += 1
            gamma_t = min(1, 2 / np.sqrt(global_step))
            if np.random.rand() < gamma_t:
                action = apply_index_policy(
                    s_list, np.random.rand(num_arms, num_states), env.arm_constraint
                )
            else:
                action = apply_index_policy(s_list, W, env.arm_constraint)
            s_dash_list, reward, _, _, info = env.step(action)
            reward_episode += reward
            traj_states.append(s_list)
            traj_actions.append(action)
            traj_next_states.append(s_dash_list)
            for arm_id, s, a, s_dash in zip(
                range(num_arms), s_list, action, s_dash_list
            ):
                Z_sa[arm_id, s, a] += 1
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
        # Get R_est using F_tilde
        ref_arm, ref_state = pick_best_ref(Wins)
        F_tilde, F_hat = enrich_F(F_tilde, F_hat, (ref_arm, ref_state), conf)
        if cfg.reward_normalized:
            F_tilde = np.clip(F_tilde, 1 / (np.e + 1), np.e / (np.e + 1))
            F_hat = np.clip(F_hat, 1 / (np.e + 1), np.e / (np.e + 1))
        else:
            F_tilde = np.clip(F_tilde, 1e-6, 1 - 1e-6)
            F_hat = np.clip(F_hat, 1e-6, 1 - 1e-6)

        # Use F estimate to compute index (policy)
        R_est = np.log(
            F_tilde[:, :, ref_arm, ref_state] / (1 - F_tilde[:, :, ref_arm, ref_state])
        )
        for s, a, s_dash in zip(traj_states, traj_actions, traj_next_states):
            for arm in range(num_arms):
                Q[arm, lambda_candidates.index(W[arm, s[arm]]), s[arm], a[arm]] = (
                    1 - 1 / (global_step)
                ) * Q[
                    arm, lambda_candidates.index(W[arm, s[arm]]), s[arm], a[arm]
                ] + 1 / (
                    global_step
                ) * (
                    R_est[arm, s[arm]]
                    - W[arm, s[arm]] * a[arm]
                    + 0.99
                    * np.max(
                        Q[arm, lambda_candidates.index(W[arm, s[arm]]), s_dash[arm], :]
                    )
                )
                for state in range(num_states):
                    W[arm, state] = lambda_candidates[
                        np.argmin(
                            [
                                np.abs(
                                    Q[arm, lambda_candidates.index(l), state, 1]
                                    - Q[arm, lambda_candidates.index(l), state, 0]
                                )
                                for l in lambda_candidates
                            ]
                        )
                    ]

        metrics["R_error"].append(np.linalg.norm(R_est - R_true))
        metrics["index_error"].append(kendalltau(W.ravel(), env.whittle_indices.ravel())[0]) # Kendall tau coeff : -1 means opposite, 0 means no correlation, 1 means same order
        metrics["F_error"].append(np.linalg.norm(F_tilde - F_true))
        wandb_log_latest(metrics)
    end_time = time.time()
    metrics["run_time"] = end_time - start_time
    return metrics
