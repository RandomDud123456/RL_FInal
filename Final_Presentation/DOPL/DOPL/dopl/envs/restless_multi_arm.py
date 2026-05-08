import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .restless_arm import RestlessArmEnv
import math
import itertools


class MultiArmRestlessEnv(gym.Env):
    """
    Environment for a multi-arm restless bandit.
    The environment consists of multiple restless arms, each of which is a Markov Decision Process.
    Info dict contains the rewards obtained from each arm.
    """

    def __init__(
        self, arm_constraint, P_list, R_list, noise_std=0.0, initial_dist=None
    ):
        super(MultiArmRestlessEnv, self).__init__()
        assert len(P_list) == len(
            R_list
        ), "Number of transition matrices and reward matrices must match"
        assert all(
            P.shape[0] == P.shape[1] and P.shape[0] == R.shape[0]
            for P, R in zip(P_list, R_list)
        ), "Number of states in the Ps and Rs must match"
        self.arm_constraint = arm_constraint
        self.arms = [
            RestlessArmEnv(P, R, noise_std, initial_dist)
            for P, R in zip(P_list, R_list)
        ]
        self.action_space = spaces.MultiDiscrete([R.shape[1] for R in R_list])
        self.observation_space = spaces.MultiDiscrete([R.shape[0] for R in R_list])
        self.action_space_combinatorial = spaces.Discrete(
            math.comb(len(self.arms), self.arm_constraint)
        )
        self.map_combinatorial_to_discrete = np.array(
            list(itertools.combinations(range(len(self.arms)), self.arm_constraint))
        )
        self.map_combinatorial_to_binary = self.generate_binary_vectors()
        self.init_exploration_phase = False

    def generate_binary_vectors(self):
        # Initialize an array to hold the binary vectors
        num_arms = len(self.arms)
        binary_vectors = np.zeros(
            (len(self.map_combinatorial_to_discrete), num_arms), dtype=int
        )
        # Convert each tuple of active arm indices to a binary vector
        for index, combination in enumerate(self.map_combinatorial_to_discrete):
            binary_vectors[index, list(combination)] = 1

        return binary_vectors

    def step(self, action):
        if not self.init_exploration_phase:
            assert (
                sum(action) <= self.arm_constraint
            ), "Action constraint on arms pulled"
        assert self.action_space.contains(action)
        rewards = []
        states = []
        for i, arm in enumerate(self.arms):
            state, reward, _, _, _ = arm.step(action[i])
            rewards.append(reward)
            states.append(state)
        assert self.observation_space.contains(states)
        ter = tru = False  # No terminal state in bandit environments
        return states, sum(rewards), ter, tru, {"arm_rewards": rewards}

    def reset(self):
        return [arm.reset() for arm in self.arms]

    def set_state(self, states):
        for i, arm in enumerate(self.arms):
            arm.set_state(states[i])

    def get_state(self):
        return [arm.get_state() for arm in self.arms]

    def render(self, mode="human"):
        for i, arm in enumerate(self.arms):
            arm.render()


if __name__ == "__main__":
    P_list = [np.array([[0.9, 0.1], [0.1, 0.9]])] * 2
    R_list = [np.array([[0.0, 1.0], [1.0, 0.0]])] * 2
    env = MultiArmRestlessEnv(1, P_list, R_list)
    print(env.action_space)
    print(env.observation_space)
    print(env.map_combinatorial_to_binary)
    print(env.reset())
