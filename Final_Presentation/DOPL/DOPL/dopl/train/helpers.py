import numpy as np
import torch
import markovianbandit as gast


def apply_index_policy(state_list, index_matrix, arm_constraint):
    """
    state_list: list of states for each arm; 1*num_arms
    index_matrix: num_arms*num_states

    return: list of actions for each arm; 1*num_arms
    """
    index_vector = index_matrix[np.arange(len(state_list)), state_list]
    largest_indices = np.argsort(index_vector)[-arm_constraint:]
    action = np.zeros(len(index_vector), dtype=int)
    action[largest_indices] = 1
    return action


def apply_nn_policy(state_list, policy_net, arm_constraint):
    """
    state_list: list of states for each arm; 1*num_arms
    policy_net: neural network

    return: list of actions for each arm; 1*num_arms
    """
    state_tensor = torch.tensor(state_list, dtype=torch.float32).unsqueeze(0)
    action_probs_tensor = policy_net(state_tensor).squeeze()
    action_probs = action_probs_tensor.detach().numpy().flatten()
    largest_indices = np.argsort(action_probs)[-arm_constraint:]
    action = np.zeros(len(action_probs), dtype=int)
    action[largest_indices] = 1
    return action, action_probs_tensor


def compute_whittle_indices(env):
    num_arms = len(env.P_list)
    whittle_indices = []
    for arm in range(num_arms):
        P0 = env.P_list[arm][:, :, 0]
        P1 = env.P_list[arm][:, :, 1]
        R0 = env.R_list[arm][:, 0]
        R1 = env.R_list[arm][:, 1]
        model = gast.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
        # assert model.is_indexable(), "Environment is not whittle indexable"
        whittle_indices.append(model.whittle_indices(discount=0.99))
    env.whittle_indices = np.array(whittle_indices, dtype=float)


def get_opt_performance(env, use_lp_solver=True):
    """Compute benchmark references for the given environment."""
    if not use_lp_solver:
        env.opt_index = None
        env.opt_index_pre_nan = None
        env.opt_cost = None
        compute_whittle_indices(env)
        return

    num_arms = len(env.P_list)
    num_states = env.P_list[0].shape[0]
    num_actions = env.R_list[0].shape[1]
    # Compute opt objective for problem
    from dopl.utils.lp_solver import compute_optimal_pyomo

    r_n_s = np.array(env.R_list)[:, :, 0]
    optimal_cost, opt_occupancy = compute_optimal_pyomo(
        # optimal_cost, opt_occupancy = compute_optimal(
        r_n_s,
        env.arm_constraint,
        np.array(env.P_list),
        num_arms,
        num_states,
        num_actions,
    )
    numerator = opt_occupancy[:, :, 1]
    denominator = opt_occupancy[:, :, 0] + opt_occupancy[:, :, 1]
    opt_index_pre_nan = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0
    )
    opt_index = np.nan_to_num(opt_index_pre_nan, nan=0.0)
    env.opt_index = opt_index
    env.opt_index_pre_nan = opt_index_pre_nan
    env.opt_cost = optimal_cost * env.H
    compute_whittle_indices(env)


def compute_F_true(env):
    num_arms = len(env.P_list)
    num_states = env.P_list[0].shape[0]
    F_true = np.zeros((num_arms, num_states, num_arms, num_states))
    for i in range(num_arms):
        for s_i in range(num_states):
            for j in range(num_arms):
                for s_j in range(num_states):
                    F_true[i, s_i, j, s_j] = np.exp(env.R_list[i][s_i, 0]) / (
                        np.exp(env.R_list[i][s_i, 0]) + np.exp(env.R_list[j][s_j, 0])
                    )
    return F_true


def pick_best_ref(W):
    ref_arm = np.random.randint(0, W.shape[0])
    ref_state = np.random.randint(0, W.shape[1])
    for arm in range(W.shape[0]):
        for state in range(W.shape[1]):
            if W[arm, state].sum() > W[ref_arm, ref_state].sum():
                ref_arm = arm
                ref_state = state
    return ref_arm, ref_state


def pick_random_ref(W):
    ref_arm = np.random.randint(0, W.shape[0])
    ref_state = np.random.randint(0, W.shape[1])
    return ref_arm, ref_state


def best_i(F_hat, reference, conf, arm, state):
    """Find best helper arm i to use in lemma 4"""
    ref_arm, ref_state = reference
    num_arms = F_hat.shape[0]
    num_states = F_hat.shape[1]
    best_arm_i, best_state_i = -1, -1
    best_conf_i = np.inf
    for arm_i in range(num_arms):
        for state_i in range(num_states):
            if arm_i == ref_arm and state_i == ref_state:
                continue
            if (
                conf[ref_arm, ref_state, arm_i, state_i]
                + conf[arm_i, state_i, arm, state]
                < best_conf_i
            ):
                best_arm_i, best_state_i = arm_i, state_i
                best_conf_i = (
                    conf[ref_arm, ref_state, arm_i, state_i]
                    + conf[arm_i, state_i, arm, state]
                )
    return best_conf_i, best_arm_i, best_state_i


def enrich_F(F_tilde, F_hat, reference, conf):
    """Use lemma 4 in DOPL paper to enrich F_tilde and F_hat: j_1 is reference, j_2 is the (arm,state) to enrich, i is the helper arm,state pair."""
    ref_arm, ref_state = reference
    num_arms = F_hat.shape[0]
    num_states = F_hat.shape[1]
    for arm in range(num_arms):
        for state in range(num_states):
            conf_inferred, best_arm_i, best_state_i = best_i(
                F_hat, reference, conf, arm, state
            )
            conf_inferred *= 1.3
            if conf_inferred < conf[ref_arm, ref_state, arm, state]:
                F_hat[ref_arm, ref_state, arm, state] = (
                    F_hat[ref_arm, ref_state, best_arm_i, best_state_i]
                    * F_hat[best_arm_i, best_state_i, arm, state]
                ) / (
                    F_hat[ref_arm, ref_state, best_arm_i, best_state_i]
                    * F_hat[best_arm_i, best_state_i, arm, state]
                    + F_hat[arm, state, best_arm_i, best_state_i]
                    * F_hat[best_arm_i, best_state_i, ref_arm, ref_state]
                )
                F_hat[arm, state, ref_arm, ref_state] = (
                    1 - F_hat[ref_arm, ref_state, arm, state]
                )
                conf[arm, state, ref_arm, ref_state] = conf[
                    ref_arm, ref_state, arm, state
                ] = conf_inferred
                F_tilde[arm, state, ref_arm, ref_state] = (
                    F_hat[arm, state, ref_arm, ref_state] + conf_inferred
                )
                F_tilde[ref_arm, ref_state, arm, state] = (
                    F_hat[ref_arm, ref_state, arm, state] + conf_inferred
                )

    return F_tilde, F_hat
