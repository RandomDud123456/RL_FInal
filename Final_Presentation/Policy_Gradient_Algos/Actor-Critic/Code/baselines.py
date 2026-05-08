"""
Baseline policies for comparison on Deadline Scheduling:
  - Random: activate M arms uniformly at random
  - Greedy-deadline: activate arms with smallest deadline first (myopic)
  - Greedy-ratio: activate arms with highest B/(D+1) first (most urgent)
"""

import numpy as np


class RandomPolicy:
    def __init__(self, N: int, M: int, seed: int = 0):
        self.N   = N
        self.M   = M
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, env=None) -> np.ndarray:
        return self.rng.choice(self.N, size=self.M, replace=False)


class GreedyDeadlinePolicy:
    """
    Activate M arms with smallest deadline D (most urgent).
    Needs raw arm states — pass env.
    """
    def __init__(self, N: int, M: int):
        self.N = N
        self.M = M

    def act(self, obs: np.ndarray, env=None) -> np.ndarray:
        if env is None:
            raise ValueError("GreedyDeadlinePolicy needs env argument")
        raw = env.state_raw_all()          # list of (D, B)
        deadlines = np.array([d for d, b in raw], dtype=float)
        # Arms with B==0 are idle — deprioritise
        for i, (d, b) in enumerate(raw):
            if b == 0:
                deadlines[i] = 1e9
        return np.argsort(deadlines)[:self.M]


class GreedyRatioPolicy:
    """
    Activate M arms with highest urgency ratio B / (D + 1).
    """
    def __init__(self, N: int, M: int):
        self.N = N
        self.M = M

    def act(self, obs: np.ndarray, env=None) -> np.ndarray:
        if env is None:
            raise ValueError("GreedyRatioPolicy needs env argument")
        raw = env.state_raw_all()
        ratios = np.array([b / (d + 1) if b > 0 else -1 for d, b in raw], dtype=float)
        return np.argsort(-ratios)[:self.M]


# ============================================================
# Exact Closed-Form Deadline Whittle Index Policy
# ============================================================

class WhittleIndexPolicy:
    """
    Exact closed-form Whittle index from:

    "Deadline Scheduling as Restless Bandits"

    State:
        (D, B)
        D = remaining deadline
        B = remaining workload

    Index:

        if D >= B:
            W(D,B) = 1 - c0

        else:
            W(D,B) =
                1 - c0
                - beta^(D-1) * (F(B-D+1) - F(B-D))

    We use:
        F(x) = penalty_coeff * x

    which is the standard linear penalty setup.
    """

    def __init__(
        self,
        N: int,
        M: int,
        c0: float = 0.0,
        beta: float = 0.99,
        penalty_coeff: float = 1.0,
    ):
        self.N = N
        self.M = M

        self.c0 = c0
        self.beta = beta
        self.penalty_coeff = penalty_coeff

    # --------------------------------------------------------
    # Penalty function F(B)
    # --------------------------------------------------------
    def F(self, x: int) -> float:
        return self.penalty_coeff * max(x, 0)

    # --------------------------------------------------------
    # Closed-form Whittle index
    # --------------------------------------------------------
    def whittle_index(self, D: int, B: int) -> float:

        # completed job
        if B == 0:
            return 0.0

        # relaxed region
        if 1 <= B <= D - 1:
            return 1.0 - self.c0

        # urgent region
        term = (
            (self.beta ** (D - 1))
            * (
                self.F(B - D + 1)
                - self.F(B - D)
            )
        )

        return 1.0 - self.c0 + term

    # --------------------------------------------------------
    # Select top-M Whittle indices
    # --------------------------------------------------------
    def act(self, obs: np.ndarray, env=None) -> np.ndarray:

        if env is None:
            raise ValueError("WhittleIndexPolicy needs env argument")

        raw = env.state_raw_all()

        indices = np.array([
            self.whittle_index(D, B)
            for D, B in raw
        ], dtype=float)

        return np.argsort(-indices)[:self.M]