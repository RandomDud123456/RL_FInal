"""
QWI: Q-Learning with Whittle Index  (Robledo et al. 2022)
----------------------------------------------------------
Designed for RMAB. Learns Whittle indices by finding the
subsidy λ(x) that makes active and passive actions equally valuable.

**Homogeneous arms** (default ``env.homogeneous=True``): one table ``Q^x(s,a)``
and index vector ``λ(s)`` shared by all arms.

**Heterogeneous arms** (``env.homogeneous=False``): separate augmented MDP per arm —
``Q_k^x(s,a)`` and ``λ_k(s)`` for arm ``k`` (shapes ``(K,N,N,2)`` and ``(K,N)``).

Key equations (per arm ``k`` in hetero mode, omit ``k`` when homogeneous):

  Q^x_{n+1}(s_n, a_n) ← (1-α) Q^x + α [ (1-a_n) λ_k(x) + a_n r + γ max_v Q^x_n(s', v) ]

  λ_k(x) ← λ_k(x) + β [ Q^x_k(x,1) - Q^x_k(x,0) ]
"""

import numpy as np
import time

from bellman_metrics import (
    whittle_augmented_bellman_rel_error,
    whittle_hetero_bellman_rel_error,
)
from metrics_common import greedy_whittle_arm, greedy_whittle_instant_reward


class QWI:
    """
    Tabular Q-learning for Whittle Index on RMAB (homogeneous or heterogeneous).

    Heterogeneous mode is selected automatically when ``env.homogeneous`` is False.
    """

    def __init__(self, env, gamma=0.9, alpha_fn=None, beta_fn=None, epsilon=1.0,
                 q_clip=None, lam_clip=None, use_double_q=True):
        self.env = env
        self.K = env.K
        self.N = env.N
        self.gamma = gamma
        self.epsilon = epsilon
        self._hetero = not bool(getattr(env, 'homogeneous', True))

        r_max = float(np.max(env.R))
        r_min = float(np.min(env.R))
        scale = r_max / max(1e-8, 1.0 - gamma)
        self._q_clip = float(q_clip) if q_clip is not None else scale * 5.0 + 10.0
        span = (r_max - r_min) / max(1e-8, 1.0 - gamma)
        self._lam_clip = float(lam_clip) if lam_clip is not None else max(scale, span) + 15.0
        self.use_double_q = bool(use_double_q)

        if alpha_fn is None:
            self.alpha_fn = lambda n: 0.1 / (1 + n // 5000)
        else:
            self.alpha_fn = alpha_fn

        if beta_fn is None:
            self.beta_fn = lambda n: 0.2 if (n % 10 == 0) else 0.0
        else:
            self.beta_fn = beta_fn

        if self._hetero:
            self.Q = np.zeros((self.K, self.N, self.N, 2))
            self.QA = np.zeros((self.K, self.N, self.N, 2))
            self.QB = np.zeros((self.K, self.N, self.N, 2))
            self.lam = np.zeros((self.K, self.N))
        else:
            self.Q = np.zeros((self.N, self.N, 2))
            self.QA = np.zeros((self.N, self.N, 2))
            self.QB = np.zeros((self.N, self.N, 2))
            self.lam = np.zeros(self.N)

        self.history = {
            'whittle_indices': [],
            'bre': [],
            'bellman_rel_error': [],
            'suboptimal_pct': [],
            'runtime': [],
            'avg_reward': [],
            'cumulative_reward': [],
            'cumulative_regret': [],
            'avg_reward_smooth': [],
            'policy_accuracy_pct': [],
            'q_delta_norm': [],
            'episodic_reward': [],
            'episodic_cumulative_regret': [],
        }
        self._rewards_since_log = []

    def _lam(self, k, x):
        return self.lam[k, x] if self._hetero else self.lam[x]

    def _set_lam(self, k, x, val):
        if self._hetero:
            self.lam[k, x] = val
        else:
            self.lam[x] = val

    def _update(self, k, s_n, a_n, r, s_next, n):
        alpha = self.alpha_fn(n)
        update_a = np.random.rand() < 0.5

        for x in range(self.N):
            lx = self._lam(k, x)
            passive_reward = (1 - a_n) * lx
            active_reward = a_n * r
            base = passive_reward + active_reward

            if self.use_double_q:
                if self._hetero:
                    if update_a:
                        a_star = int(np.argmax(self.QA[k, x, s_next]))
                        target = base + self.gamma * self.QB[k, x, s_next, a_star]
                        new_q = (1 - alpha) * self.QA[k, x, s_n, a_n] + alpha * target
                        self.QA[k, x, s_n, a_n] = np.clip(new_q, -self._q_clip, self._q_clip)
                    else:
                        a_star = int(np.argmax(self.QB[k, x, s_next]))
                        target = base + self.gamma * self.QA[k, x, s_next, a_star]
                        new_q = (1 - alpha) * self.QB[k, x, s_n, a_n] + alpha * target
                        self.QB[k, x, s_n, a_n] = np.clip(new_q, -self._q_clip, self._q_clip)
                else:
                    if update_a:
                        a_star = int(np.argmax(self.QA[x, s_next]))
                        target = base + self.gamma * self.QB[x, s_next, a_star]
                        new_q = (1 - alpha) * self.QA[x, s_n, a_n] + alpha * target
                        self.QA[x, s_n, a_n] = np.clip(new_q, -self._q_clip, self._q_clip)
                    else:
                        a_star = int(np.argmax(self.QB[x, s_next]))
                        target = base + self.gamma * self.QA[x, s_next, a_star]
                        new_q = (1 - alpha) * self.QB[x, s_n, a_n] + alpha * target
                        self.QB[x, s_n, a_n] = np.clip(new_q, -self._q_clip, self._q_clip)
            else:
                if self._hetero:
                    q_next = 0.5 * (self.Q[k, x, s_next, 0] + self.Q[k, x, s_next, 1])
                    target = base + self.gamma * q_next
                    new_q = (1 - alpha) * self.Q[k, x, s_n, a_n] + alpha * target
                    self.Q[k, x, s_n, a_n] = np.clip(new_q, -self._q_clip, self._q_clip)
                else:
                    q_next = 0.5 * (self.Q[x, s_next, 0] + self.Q[x, s_next, 1])
                    target = base + self.gamma * q_next
                    new_q = (1 - alpha) * self.Q[x, s_n, a_n] + alpha * target
                    self.Q[x, s_n, a_n] = np.clip(new_q, -self._q_clip, self._q_clip)

    def _step_lambda(self, n):
        beta = self.beta_fn(n)
        if beta <= 0:
            return
        if self._hetero:
            for k in range(self.K):
                for x in range(self.N):
                    if self.use_double_q:
                        qd = 0.5 * (
                            (self.QA[k, x, x, 1] - self.QA[k, x, x, 0])
                            + (self.QB[k, x, x, 1] - self.QB[k, x, x, 0])
                        )
                    else:
                        qd = self.Q[k, x, x, 1] - self.Q[k, x, x, 0]
                    nv = self._lam(k, x) + beta * qd
                    self._set_lam(k, x, np.clip(nv, -self._lam_clip, self._lam_clip))
        else:
            for x in range(self.N):
                if self.use_double_q:
                    qd = 0.5 * (
                        (self.QA[x, x, 1] - self.QA[x, x, 0])
                        + (self.QB[x, x, 1] - self.QB[x, x, 0])
                    )
                else:
                    qd = self.Q[x, x, 1] - self.Q[x, x, 0]
                nv = self.lam[x] + beta * qd
                self.lam[x] = np.clip(nv, -self._lam_clip, self._lam_clip)

    def _select_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.K)
        if self._hetero:
            indices = [self.lam[i, self.env.states[i]] for i in range(self.K)]
        else:
            indices = [self.lam[self.env.states[i]] for i in range(self.K)]
        return int(np.argmax(indices))

    def train(self, n_steps=50000, true_indices=None, optimal_policy=None,
              log_every=500, epsilon_decay=None, trace_episode_len=None):
        states = self.env.reset()
        start = time.time()
        cum_r = 0.0
        cum_reg = 0.0
        oracle_hits = 0
        ema_smooth = None
        ema_beta = 0.12
        q_prev = None
        tel = int(trace_episode_len) if trace_episode_len else 0
        ep_acc = 0.0

        for n in range(1, n_steps + 1):
            if epsilon_decay is not None:
                self.epsilon = epsilon_decay(n, self.epsilon)

            arm = self._select_arm()
            next_states, reward = self.env.step(arm)
            if true_indices is not None:
                r_star = greedy_whittle_instant_reward(self.env, true_indices, states)
                cum_reg += r_star - float(reward)
                if arm == greedy_whittle_arm(self.env, true_indices, states):
                    oracle_hits += 1
            cum_r += float(reward)
            self._rewards_since_log.append(reward)
            if tel > 0:
                ep_acc += float(reward)

            s_n = states[arm]
            s_next = next_states[arm]

            if self._hetero:
                self._update(arm, s_n, 1, reward, s_next, n)
                for i in range(self.K):
                    if i != arm:
                        self._update(i, states[i], 0, 0.0, next_states[i], n)
            else:
                self._update(0, s_n, 1, reward, s_next, n)
                for i in range(self.K):
                    if i != arm:
                        self._update(0, states[i], 0, 0.0, next_states[i], n)

            self._step_lambda(n)
            states = next_states

            if tel > 0 and n % tel == 0:
                self.history['episodic_reward'].append(float(ep_acc))
                self.history['episodic_cumulative_regret'].append(float(cum_reg))
                ep_acc = 0.0

            if n % log_every == 0:
                elapsed = time.time() - start
                self.history['whittle_indices'].append(self.lam.copy())
                self.history['runtime'].append(elapsed)
                if self._rewards_since_log:
                    win_mean = float(np.mean(self._rewards_since_log))
                    self._rewards_since_log = []
                else:
                    win_mean = float("nan")
                self.history['avg_reward'].append(win_mean)
                if np.isfinite(win_mean):
                    ema_smooth = (
                        win_mean if ema_smooth is None
                        else (1 - ema_beta) * ema_smooth + ema_beta * win_mean
                    )
                    self.history['avg_reward_smooth'].append(float(ema_smooth))

                self.history['cumulative_reward'].append(float(cum_r))
                if true_indices is not None:
                    self.history['cumulative_regret'].append(float(cum_reg))
                    self.history['policy_accuracy_pct'].append(
                        100.0 * float(oracle_hits) / float(log_every))
                    oracle_hits = 0
                else:
                    self.history['cumulative_regret'].append(float('nan'))
                    self.history['policy_accuracy_pct'].append(float('nan'))

                v = (
                    np.concatenate([self.QA.ravel(), self.QB.ravel()])
                    if self.use_double_q else self.Q.ravel()
                )
                if q_prev is None:
                    self.history['q_delta_norm'].append(0.0)
                else:
                    self.history['q_delta_norm'].append(float(np.linalg.norm(v - q_prev)))
                q_prev = v.copy()

                if true_indices is not None:
                    bre = self._compute_bre(true_indices)
                    self.history['bre'].append(bre)

                if self._hetero:
                    br = whittle_hetero_bellman_rel_error(
                        self.env, self.gamma, self.lam, self.use_double_q,
                        QA_K=self.QA if self.use_double_q else None,
                        QB_K=self.QB if self.use_double_q else None,
                        Q_K=self.Q if not self.use_double_q else None,
                    )
                else:
                    br = whittle_augmented_bellman_rel_error(
                        self.env, self.gamma, self.lam,
                        Q=self.Q if not self.use_double_q else None,
                        use_double_q=self.use_double_q,
                        QA=self.QA if self.use_double_q else None,
                        QB=self.QB if self.use_double_q else None,
                    )
                self.history['bellman_rel_error'].append(br)

                if optimal_policy is not None:
                    pct = self._compute_suboptimal_pct(states, optimal_policy, n)
                    self.history['suboptimal_pct'].append(pct)

        if tel > 0 and n_steps % tel != 0 and ep_acc != 0.0:
            self.history['episodic_reward'].append(float(ep_acc))
            self.history['episodic_cumulative_regret'].append(float(cum_reg))

        return self.lam.copy()

    def _compute_bre(self, true_indices):
        t = np.asarray(true_indices, dtype=float)
        if t.ndim == 1:
            return float(np.mean(np.abs(self.lam - t)))
        return float(np.mean(np.abs(self.lam - t)))

    def _compute_suboptimal_pct(self, states, optimal_policy, n):
        chosen = self._select_arm()
        optimal = optimal_policy(states)
        return float(chosen != optimal)

    def get_indices(self):
        return self.lam.copy()
