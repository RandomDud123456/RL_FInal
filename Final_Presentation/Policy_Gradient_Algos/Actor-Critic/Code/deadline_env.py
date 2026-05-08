"""
Deadline Scheduling Environment
Exact match to NeurWIN paper (Section 5.2):
  - EV charging station with N spots, M vehicles can be charged per round
  - State of arm i: (D, B) where D = deadline (rounds remaining), B = remaining charge
  - D <= 12, B <= 9  →  state space = 120 per arm
  - Reward per unit electricity delivered: (1 - c)
  - Penalty for unfulfilled charge: F(B) = B (linear, as in [34])
  - New vehicle arrives when spot is vacated
"""

import numpy as np


# ── Problem constants (matching NeurWIN paper) ──────────────────────────────
MAX_DEADLINE = 12   # D ∈ {1, …, 12}
MAX_BATTERY  = 9    # B ∈ {0, …, 9}   (0 = fully charged / empty spot)
REWARD_RATE  = 1.0  # reward per unit charge delivered  (set c=0 for simplicity)
COST         = 0.0  # c in (1-c)
PENALTY_COEF = 1.0  # F(B) = PENALTY_COEF * B


def _penalty(b: int) -> float:
    return PENALTY_COEF * b


def _reward_per_unit() -> float:
    return 1.0 - COST


class DeadlineArm:
    """
    Single EV-charging arm.
    State: (D, B)  — integers.
    D: rounds until vehicle departs.  B: units of charge still needed.
    """

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()
        self.D = 0
        self.B = 0
        self._spawn_vehicle()

    # ── internal helpers ────────────────────────────────────────────────────
    def _spawn_vehicle(self):
        """A new vehicle occupies the spot."""
        self.D = self.rng.integers(1, MAX_DEADLINE + 1)
        self.B = self.rng.integers(1, MAX_BATTERY  + 1)

    def state(self) -> np.ndarray:
        """Return normalised state vector [D/D_max, B/B_max]."""
        return np.array([self.D / MAX_DEADLINE, self.B / MAX_BATTERY], dtype=np.float32)

    def state_raw(self) -> tuple[int, int]:
        return (self.D, self.B)

    # ── step ────────────────────────────────────────────────────────────────
    def step(self, active: bool) -> float:
        """
        Transition the arm for one round.
        active=True  → charge the vehicle (deliver 1 unit if B > 0)
        Returns the reward for this round.
        """
        reward = 0.0

        if self.B == 0:
            # Spot is idle — no vehicle. Spawn one.
            self._spawn_vehicle()
            return 0.0

        if active and self.B > 0:
            # Deliver 1 unit of charge
            self.B -= 1
            reward += _reward_per_unit()

        # Decrement deadline
        self.D -= 1

        if self.D == 0:
            # Vehicle departs — pay penalty for unfulfilled charge
            reward -= _penalty(self.B)
            # Spawn next vehicle
            self._spawn_vehicle()

        return reward

    def reset_to(self, d: int, b: int):
        self.D = d
        self.B = b


class DeadlineSchedulingEnv:
    """
    Multi-arm deadline scheduling environment.
    N arms (charging spots), M can be activated per round.

    Compatible with the NeurWIN benchmark in Section 5.2.
    """

    def __init__(self, N: int, M: int, seed: int = 42):
        assert M <= N
        self.N   = N
        self.M   = M
        self.rng = np.random.default_rng(seed)
        self.arms = [DeadlineArm(np.random.default_rng(seed + i)) for i in range(N)]

    def reset(self) -> np.ndarray:
        """Reset all arms; return stacked states [N, 2]."""
        for arm in self.arms:
            arm._spawn_vehicle()
        return self._obs()

    def _obs(self) -> np.ndarray:
        return np.stack([arm.state() for arm in self.arms])   # [N, 2]

    def step(self, action_indices: np.ndarray):
        """
        action_indices: array of M arm indices to activate.
        Returns: (obs, total_reward, arm_rewards)
        """
        active_set = set(action_indices.tolist())
        arm_rewards = np.zeros(self.N, dtype=np.float32)
        for i, arm in enumerate(self.arms):
            arm_rewards[i] = arm.step(i in active_set)
        total_reward = arm_rewards.sum()
        return self._obs(), total_reward, arm_rewards

    def state_raw_all(self) -> list[tuple[int, int]]:
        return [arm.state_raw() for arm in self.arms]
