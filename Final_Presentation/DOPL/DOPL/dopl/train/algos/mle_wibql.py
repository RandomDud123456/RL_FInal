from tqdm import tqdm
import numpy as np
from dopl.utils import wandb_log_latest, normalize_matrix
from dopl.registry import register_training_function
from dopl.train.algos.bradley_terry import mle_bradley_terry
from dopl.train.helpers import apply_index_policy
import time
from scipy.stats import kendalltau

def func(Q, num_states):
    """input : Q dimensionality [ states, actions]
    input : num_states scalar (d in Borkar paper)
    output : func(Q) scalar"""
    return 1 / (2 * num_states) * np.sum(Q)


def a_seq(n, step_size_params):
    n += 1
    C = step_size_params["C"]
    D = step_size_params["D"]
    return C / np.ceil(n / D)


def b_seq(n, step_size_params):
    n += 1
    C_dash = step_size_params["C_dash"]
    D_dash = step_size_params["D_dash"]
    return C_dash / (1 + np.ceil(n * np.log(n) / D_dash))


@register_training_function("mle_wibql")
def train(env, cfg):
    start_time = time.time()
    K = cfg["K"]
    epsilon = cfg["epsilon"]
    R_true = np.array(env.R_list)[:, :, 0]

    num_arms = len(env.P_list)
    num_states = env.P_list[0].shape[0]
    num_actions = env.R_list[0].shape[1]

    R_est = np.ones((num_arms, num_states)) * 0.5
    Q = np.random.rand(
        num_arms,
        num_states,
        num_actions,
        num_states,
    )
    W = np.random.rand(num_arms, num_states)
    Z_sa = np.zeros((num_arms, num_states, num_actions))

    metrics = {
        "reward": [],
        "index_error": [],
        "R_error": [],
        "run_time": 0,
    }

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
            if np.random.rand() < epsilon:
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
                battle_data.append(
                    [winner, s_dash_list[winner], loser, s_dash_list[loser]]
                )
            s_list = s_dash_list

        metrics["reward"].append(reward_episode)
        # Update R_est using Bradley-Terry model
        R_est = mle_bradley_terry(np.array(battle_data), R_est)
        for arm in range(num_arms):
            for state in range(num_states):  # k_hat in Borkar paper
                # Update Q values based on traj data
                for s, a, s_dash in zip(traj_states, traj_actions, traj_next_states):
                    Q[arm, s[arm], a[arm], state] = Q[
                        arm, s[arm], a[arm], state
                    ] + a_seq(Z_sa[arm, s[arm], a[arm]], cfg["step_size_params"]) * (
                        (1 - a[arm]) * (R_est[arm, s[arm]] + W[arm, state])
                        + a[arm] * R_est[arm, s[arm]]
                        + np.max(Q[arm, s_dash[arm], :, state])
                        - func(Q[arm, :, :, state], num_states)
                        - Q[arm, s[arm], a[arm], state]
                    )
                # Update W value for state
                W[arm, state] = W[arm, state] + (b_seq(k,cfg["step_size_params"])) * (
                    Q[arm, state, 1, state] - Q[arm, state, 0, state]
                )
        metrics["R_error"].append(np.linalg.norm(R_est - R_true))
        metrics["index_error"].append(kendalltau(W.ravel(), env.whittle_indices.ravel())[0]) # Kendall tau coeff : -1 means opposite, 0 means no correlation, 1 means same order
        wandb_log_latest(metrics)
    end_time = time.time()
    metrics["run_time"] = end_time - start_time
    return metrics
