from tqdm import tqdm
import numpy as np
from dopl.utils import wandb_log_latest
from dopl.registry import register_training_function
import torch
from dopl.train.algos.bradley_terry import mle_bradley_terry
from dopl.train.helpers import apply_index_policy
import time
from scipy.stats import kendalltau

def a_seq(const_size):
    return const_size


@register_training_function("mle_qwic")
def train(env, cfg):
    start_time = time.time()
    K = cfg["K"]
    R_true = np.array(env.R_list)[:, :, 0]

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
    Z_sa = np.zeros((num_arms, num_states, num_actions))

    metrics = {
        "reward": [],
        "index_error": [],
        "R_error": [],
        "run_time": 0,
    }
    global_step = 0
    # battle_data = []
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
            for record in info["duelling_results"]:
                winner, loser = record
                battle_data.append(
                    [winner, s_dash_list[winner], loser, s_dash_list[loser]]
                )
            s_list = s_dash_list

        metrics["reward"].append(reward_episode)
        # Update R_est using Bradley-Terry model
        R_est = mle_bradley_terry(np.array(battle_data), R_est)
        for s, a, s_dash in zip(traj_states, traj_actions, traj_next_states):
            for arm in range(num_arms):
                Q[arm, lambda_candidates.index(W[arm, s[arm]]), s[arm], a[arm]] = (
                    1 - a_seq(cfg["step_size_params"]["const_step"])
                ) * Q[
                    arm, lambda_candidates.index(W[arm, s[arm]]), s[arm], a[arm]
                ] + a_seq(
                    cfg["step_size_params"]["const_step"]
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
        wandb_log_latest(metrics)
    end_time = time.time()
    metrics["run_time"] = end_time - start_time
    return metrics
