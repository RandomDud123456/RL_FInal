import numpy as np
import gymnasium as gym
from gymnasium import spaces


class RestlessArmEnv(gym.Env):
    """A restless arm modeled as a Markov Decision Process."""

    def __init__(self, P, R, noise_std=0.0, initial_dist=None):
        super(RestlessArmEnv, self).__init__()
        self.P = P  # Transition matrix of size |S|x|S|x|A|
        self.R = R  # Reward matrix of size |S|x|A|
        self.n_states, self.n_actions = R.shape
        self.action_space = spaces.Discrete(self.n_actions)
        self.state_space = spaces.Discrete(self.n_states)
        self.initial_dist = initial_dist
        self.noise_std = noise_std  # Standard deviation of the Gaussian noise

    def step(self, action):
        action = int(action)
        assert self.action_space.contains(action)

        noise = np.random.normal(0, self.noise_std)  # AWGN for the reward
        reward = self.R[self.current_state][action] + noise
        reward = np.clip(reward, 0, 1)  # clip reward to [0, 1]

        self.current_state = np.random.choice(
            self.n_states, p=self.P[self.current_state, :, action]
        )
        ter = tru = False  # Bandit environments typically do not have a terminal state
        return self.current_state, reward, ter, tru, {}

    def reset(self):
        self.current_state = (
            np.random.choice(self.n_states, p=self.initial_dist)
            if self.initial_dist is not None
            else np.random.randint(self.n_states)
        )
        return self.current_state

    def set_state(self, state):
        self.current_state = state

    def get_state(self):
        return self.current_state

    def render(self, mode="human"):
        print(f"State: {self.current_state}")
