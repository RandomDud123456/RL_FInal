# app marketing environment
import numpy as np


def main():
    # Define the number of states and actions
    n_states = 4
    n_actions = 2

    # Initialize and define transition and reward matrices for each arm separately
    P_arm1, R_arm1 = initialize_arm1(n_states, n_actions)
    # P_arm2, R_arm2 = initialize_arm2(n_states, n_actions)

    # Verify the transition matrices
    verify_transitions(P_arm1, "Arm 1")
    # verify_transitions(P_arm2, "Arm 2")

    # Save matrices to files
    save_matrices(P_arm1, R_arm1, "arm_type_1")
    # save_matrices(P_arm2, R_arm2, "arm_type_2")


def initialize_arm1(n_states, n_actions):
    # Transition matrices for arm 1
    P = np.zeros((n_states, n_states, n_actions))
    P[:, :, 0] = [
        [0.7, 0.1, 0.1, 0.1],
        [0.5, 0.3, 0.1, 0.1],
        [0.2, 0.4, 0.3, 0.1],
        [0.1, 0.2, 0.2, 0.5],
    ]  # Action 0
    P[:, :, 1] = [
        [0.1, 0.1, 0.7, 0.1],
        [0.1, 0.1, 0.1, 0.7],
        [0.1, 0.1, 0.1, 0.7],
        [0.05, 0.05, 0.05, 0.85],
    ]  # Action 1

    # Reward matrix for arm 1
    R = np.zeros((n_states, n_actions))
    for state_index in range(n_states):
        R[state_index, :] = state_index / (n_states - 1)

    return P, R


# def initialize_arm2(n_states, n_actions):
#     # Transition matrices for arm 2
#     P = np.zeros((n_states, n_states, n_actions))
#     P[:, :, 0] = [
#         [0.7427, 0.0741, 0.1832],  # Action 0
#         [0.3399, 0.1634, 0.4967],
#         [0.2323, 0.1020, 0.6657],
#     ]
#     P[:, :, 1] = [
#         [0.1427, 0.3741, 0.4832],  # Action 1
#         [0.1399, 0.1, 0.7601],
#         [0.1323, 0.1, 0.7677],
#     ]

#     # Reward matrix for arm 2
#     R = np.zeros((n_states, n_actions))
#     for state_index in range(n_states):
#         R[state_index, :] = state_index / (n_states - 1)

#     return P, R


def verify_transitions(P, arm_description):
    row_sums = np.sum(P, axis=1)
    if not np.allclose(row_sums, np.ones((P.shape[0], P.shape[2])), atol=1e-8):
        print(f"{arm_description}: Row sums not 1 - ERROR")
        print("Row sums:", row_sums)
    else:
        print(f"{arm_description}: Rows sum to 1 - OK")


def save_matrices(P, R, arm_type):
    np.save(f"app_marketing_{arm_type}_transitions.npy", P)
    np.save(f"app_marketing_{arm_type}_rewards.npy", R)
    print(f"Matrices for {arm_type} saved successfully.")


if __name__ == "__main__":
    main()
