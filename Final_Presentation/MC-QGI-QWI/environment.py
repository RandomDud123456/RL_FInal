"""
RMAB Environment
----------------
A Restless Multi-Armed Bandit environment where:
- Each arm is a Markov chain
- Active arms (pulled) transition via P_active
- Passive arms (not pulled) ALSO transition via P_passive  <-- key RMAB feature
- Rewards are received only on active pulls
"""

import numpy as np


class RMABEnvironment:
    """
    Restless Multi-Armed Bandit Environment.

    Parameters
    ----------
    K        : number of arms
    N        : number of states per arm
    P_active : active transition matrix  shape (K, N, N) or (N, N) if homogeneous
    P_passive: passive transition matrix shape (K, N, N) or (N, N) if homogeneous
    R        : reward matrix             shape (K, N)    or (N,)   if homogeneous
    homogeneous : if True, all arms share same P_active, P_passive, R
    """

    def __init__(self, K, N, P_active, P_passive, R, homogeneous=True, restless=True):
        self.K = K
        self.N = N
        self.homogeneous = homogeneous
        self.restless = bool(restless)

        # Expand to per-arm matrices if homogeneous
        if homogeneous:
            self.P_active  = np.stack([P_active]  * K)   # (K, N, N)
            self.P_passive = np.stack([P_passive] * K)   # (K, N, N)
            self.R         = np.stack([R]         * K)   # (K, N)
        else:
            self.P_active  = P_active   # already (K, N, N)
            self.P_passive = P_passive
            self.R         = R

        self.states = np.zeros(K, dtype=int)  # current state of each arm

    def reset(self):
        """Reset all arms to random initial states."""
        self.states = np.random.randint(0, self.N, size=self.K)
        return self.states.copy()

    def step(self, arm_pulled):
        """
        Pull one arm.

        - Restless (RMAB): all other arms transition via ``P_passive``.
        - Rested: all other arms stay in their current state.

        Returns
        -------
        next_states : new state of every arm
        reward      : reward from the pulled arm
        """
        reward = self.R[arm_pulled, self.states[arm_pulled]]
        next_states = np.zeros(self.K, dtype=int)

        for i in range(self.K):
            s = self.states[i]
            if i == arm_pulled:
                # Active transition
                next_states[i] = np.random.choice(self.N, p=self.P_active[i, s])
            else:
                if self.restless:
                    # Passive transition  <-- this is what makes it RMAB not MAB
                    next_states[i] = np.random.choice(self.N, p=self.P_passive[i, s])
                else:
                    # Rested arms don't evolve when not pulled
                    next_states[i] = int(s)

        self.states = next_states
        return next_states.copy(), reward

    def step_concurrent(self, actions):
        """
        Activate a subset of arms in parallel (budget = number of ones in ``actions``).

        Each arm ``i`` uses ``P_active[i]`` if ``actions[i]==1``, else ``P_passive[i]``.
        Reward is ``R[i, s_i]`` on active arms only (passive immediate reward 0), matching
        the standard RMAB reward structure in ``step(arm_pulled)``.

        For ``restless=False`` (rested), passive arms do not transition.

        Parameters
        ----------
        actions : (K,) int/bool — binary activation vector

        Returns
        -------
        next_states, rewards (K,), done (always False here)
        """
        actions = np.asarray(actions, dtype=int).reshape(self.K)
        rewards = np.zeros(self.K, dtype=float)
        next_states = np.zeros(self.K, dtype=int)

        for i in range(self.K):
            s = int(self.states[i])
            if int(actions[i]) == 1:
                Prow = self.P_active[i, s]
                rewards[i] = float(self.R[i, s])
            else:
                Prow = self.P_passive[i, s]
                rewards[i] = 0.0
            if int(actions[i]) == 0 and not self.restless:
                next_states[i] = s
            else:
                next_states[i] = int(np.random.choice(self.N, p=Prow))

        self.states = next_states
        return next_states.copy(), rewards, False

    def clone_with_restless(self, restless: bool):
        """
        Same ``P_active``, ``P_passive``, ``R``, and dimensions; only ``restless`` changes.

        Use **restless=True** for Whittle / MC-QWI (passive arms evolve with ``P_passive``)
        and **restless=False** for rested / MC-QGI alignment (passive arms frozen), per the
        usual QWI-vs-QGI diagram split.
        """
        restless = bool(restless)
        if self.homogeneous:
            return RMABEnvironment(
                self.K,
                self.N,
                np.array(self.P_active[0], copy=True),
                np.array(self.P_passive[0], copy=True),
                np.array(self.R[0], copy=True),
                homogeneous=True,
                restless=restless,
            )
        return RMABEnvironment(
            self.K,
            self.N,
            np.array(self.P_active, copy=True),
            np.array(self.P_passive, copy=True),
            np.array(self.R, copy=True),
            homogeneous=False,
            restless=restless,
        )

    # ------------------------------------------------------------------
    # Factory methods for common benchmark environments
    # ------------------------------------------------------------------

    @classmethod
    def make_random(cls, K=5, N=10, seed=42, restless=True):
        """Random Dirichlet transition matrices — standard benchmark."""
        rng = np.random.RandomState(seed)

        def random_stochastic(n):
            M = rng.dirichlet(np.ones(n), size=n)
            return M

        P_active  = random_stochastic(N)
        P_passive = random_stochastic(N)
        R = rng.uniform(0, 1, size=N)
        return cls(K, N, P_active, P_passive, R, homogeneous=True, restless=restless)

    @classmethod
    def make_toy(cls, K=5, N=5, restless=True):
        """
        Toy example from the QGI paper (modified for RMAB), generalized to any N ≥ 2.

        Active: from state s < N−1, jump to 0 (prob 0.3) or advance to s+1 (0.7);
        from N−1, jump to 0 (0.3) or stay (0.7) — same pattern as the original N=5 matrices.

        Passive kernel is defined for restless dynamics (drift). If ``restless=False``,
        passive rows are still stored but **frozen passive** semantics apply in ``step``.
        """
        if N < 2:
            raise ValueError('make_toy requires N >= 2')

        P_active = np.zeros((N, N))
        for s in range(N - 1):
            P_active[s, 0] = 0.3
            P_active[s, s + 1] = 0.7
        P_active[N - 1, 0] = 0.3
        P_active[N - 1, N - 1] = 0.7

        P_passive = np.zeros((N, N))
        P_passive[0, 0] = 0.8
        P_passive[0, 1] = 0.2
        for s in range(1, N):
            P_passive[s, s - 1] = 0.5
            P_passive[s, s] = 0.5

        R = np.array([0.9**s + 1.0 for s in range(N)], dtype=float)
        return cls(K, N, P_active, P_passive, R, homogeneous=True, restless=restless)

    @classmethod
    def make_hetero_toy(cls, K=4, N=5, seed=0, delta=0.1, restless=True):
        """
        Same skeleton as ``make_toy``, but **each arm** gets slightly different
        row-stochastic ``P_active``, ``P_passive`` and reward vector ``R``.
        Returns ``homogeneous=False`` with shapes ``(K,N,N)`` and ``(K,N)``.
        """
        if N < 2:
            raise ValueError('make_hetero_toy requires N >= 2')
        rng = np.random.RandomState(seed)

        def toy_active():
            P = np.zeros((N, N))
            for s in range(N - 1):
                P[s, 0] = 0.3
                P[s, s + 1] = 0.7
            P[N - 1, 0] = 0.3
            P[N - 1, N - 1] = 0.7
            return P

        def toy_passive():
            P = np.zeros((N, N))
            P[0, 0] = 0.8
            P[0, 1] = 0.2
            for s in range(1, N):
                P[s, s - 1] = 0.5
                P[s, s] = 0.5
            return P

        P_list, Q_list, R_list = [], [], []
        for k in range(K):
            Pa = toy_active().copy()
            Pp = toy_passive().copy()
            for s in range(N):
                mix_a = rng.dirichlet(np.ones(N)) * delta + Pa[s] * (1.0 - delta)
                Pa[s] = mix_a / mix_a.sum()
                mix_p = rng.dirichlet(np.ones(N)) * delta + Pp[s] * (1.0 - delta)
                Pp[s] = mix_p / mix_p.sum()
            Rk = np.array([0.9 ** s + 1.0 for s in range(N)], dtype=float)
            Rk *= 1.0 + rng.uniform(-delta, delta, size=N)
            P_list.append(Pa)
            Q_list.append(Pp)
            R_list.append(Rk)

        P_active = np.stack(P_list, axis=0)
        P_passive = np.stack(Q_list, axis=0)
        R = np.stack(R_list, axis=0)
        return cls(K, N, P_active, P_passive, R, homogeneous=False, restless=restless)

    @classmethod
    def make_wireless(cls, K=10, N=5, seed=0, restless=True):
        """
        Wireless scheduling benchmark.
        Channel quality evolves as a Markov chain.
        Active: transmit, get reward = channel state quality.
        Passive: channel evolves independently.
        """
        rng = np.random.RandomState(seed)
        # Gilbert-Elliott like channel model
        p = rng.uniform(0.1, 0.4, size=(N, N))
        P_active  = p / p.sum(axis=1, keepdims=True)
        q = rng.uniform(0.1, 0.4, size=(N, N))
        P_passive = q / q.sum(axis=1, keepdims=True)
        R = np.arange(1, N+1, dtype=float) / N  # higher state = better channel
        return cls(K, N, P_active, P_passive, R, homogeneous=True, restless=restless)
