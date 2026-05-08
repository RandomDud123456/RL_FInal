import numpy as np
from .restless_multi_arm import MultiArmRestlessEnv


class MultiArmRestlessDuellingEnv(MultiArmRestlessEnv):
    """
    Environment for a duelling multi-arm restless bandit.
    Info dict contains the results of each duel.
    """

    def __init__(self, arm_constraint, P_list, R_list, noise_std=0.0,initial_dist=None):
        super().__init__(arm_constraint, P_list, R_list, noise_std,initial_dist)
        self.P_list = P_list
        self.R_list = R_list
        # Define the data type for storing duel results
        self.dtype = [("winner", int), ("loser", int)]

    def step(self, action):
        # Call the original RMAB step method
        states, reward, ter, tru, info = super().step(action)
        arm_rewards = info["arm_rewards"]

        ## Perform the ```arm_constraint```` choose 2 duels
        # Filter active arms
        active_arms = np.where(action > 0)[0]
        n_active = len(active_arms)

        # Compute all duels among active arms
        duelling_results = []
        for i in range(n_active):
            for j in range(i + 1, n_active):
                idx_i = active_arms[i]
                idx_j = active_arms[j]
                prob_i_wins = np.exp(arm_rewards[idx_i]) / (
                    np.exp(arm_rewards[idx_i]) + np.exp(arm_rewards[idx_j])
                )
                if np.random.rand() < prob_i_wins:
                    duelling_results.append((idx_i, idx_j))  # arm i wins
                else:
                    duelling_results.append((idx_j, idx_i))  # arm j wins

        duelling_results = np.array(duelling_results, dtype=self.dtype)

        return (
            states,
            reward,
            ter,
            tru,
            {"arm_rewards": arm_rewards, "duelling_results": duelling_results},
        )
