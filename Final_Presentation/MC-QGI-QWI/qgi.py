"""
QGI: Q-Learning for Gittins Index  (Dhankhar et al. 2024)
----------------------------------------------------------
Adapted for RMAB: updates use the **pulled arm's** active transition only.

**Homogeneous** ``(env.homogeneous=True)``: ``Q`` (N,N), ``M`` (N).

**Heterogeneous** ``(env.homogeneous=False)``: per-arm retirement tables
``Q_k^x(s)`` and ``M_k(x)`` with shapes ``(K,N,N)`` and ``(K,N)``.
"""

import numpy as np
import time

from bellman_metrics import (
    gittins_active_bellman_rel_error,
    gittins_hetero_bellman_rel_error,
)
from metrics_common import greedy_whittle_arm, greedy_whittle_instant_reward


class QGI:
    """Tabular Q-learning for Gittins index (homogeneous or heterogeneous arms)."""

    def __init__(self, env, gamma=0.9, alpha_fn=None, beta_fn=None, epsilon=1.0):
        self.env = env
        self.K = env.K
        self.N = env.N
        self.gamma = gamma
        self.epsilon = epsilon
        self._hetero = not bool(getattr(env, 'homogeneous', True))

        if alpha_fn is None:
            self.alpha_fn = lambda n: 0.6 / (1 + n // 5000)
        else:
            self.alpha_fn = alpha_fn

        if beta_fn is None:
            self.beta_fn = lambda n: 0.4 if (n % 5 == 0) else 0.0
        else:
            self.beta_fn = beta_fn

        if self._hetero:
            self.Q = np.zeros((self.K, self.N, self.N))
            self.M = np.zeros((self.K, self.N))
        else:
            self.Q = np.zeros((self.N, self.N))
            self.M = np.zeros(self.N)

        self.history = {
            'indices': [],
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

    def _update(self, k, s_n, r, s_next, n):
        alpha = self.alpha_fn(n)
        beta = self.beta_fn(n)

        for x in range(self.N):
            if self._hetero:
                target = r + self.gamma * max(self.Q[k, x, s_next], self.M[k, x])
                self.Q[k, x, s_n] = (1 - alpha) * self.Q[k, x, s_n] + alpha * target
                if beta > 0:
                    self.M[k, x] = self.M[k, x] + beta * (self.Q[k, x, x] - self.M[k, x])
            else:
                target = r + self.gamma * max(self.Q[x, s_next], self.M[x])
                self.Q[x, s_n] = (1 - alpha) * self.Q[x, s_n] + alpha * target
                if beta > 0:
                    self.M[x] = self.M[x] + beta * (self.Q[x, x] - self.M[x])

    def _select_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.K)
        if self._hetero:
            indices = [self.M[i, self.env.states[i]] for i in range(self.K)]
        else:
            indices = [self.M[self.env.states[i]] for i in range(self.K)]
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
            self._update(arm, s_n, reward, s_next, n)
            states = next_states

            if tel > 0 and n % tel == 0:
                self.history['episodic_reward'].append(float(ep_acc))
                self.history['episodic_cumulative_regret'].append(float(cum_reg))
                ep_acc = 0.0

            if n % log_every == 0:
                elapsed = time.time() - start
                self.history['indices'].append(self.M.copy())
                self.history['whittle_indices'].append((1.0 - self.gamma) * self.M.copy())
                self.history['runtime'].append(elapsed)
                # Always log one scalar per checkpoint (same length as ``bre`` / Bellman rows).
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

                v = self.Q.ravel()
                if q_prev is None:
                    self.history['q_delta_norm'].append(0.0)
                else:
                    self.history['q_delta_norm'].append(float(np.linalg.norm(v - q_prev)))
                q_prev = v.copy()

                if true_indices is not None:
                    learned = (1.0 - self.gamma) * self.M
                    t = np.asarray(true_indices, dtype=float)
                    bre = float(np.mean(np.abs(learned - t)))
                    self.history['bre'].append(bre)

                if self._hetero:
                    br = gittins_hetero_bellman_rel_error(self.env, self.gamma, self.M, self.Q)
                else:
                    br = gittins_active_bellman_rel_error(self.env, self.gamma, self.M, self.Q)
                self.history['bellman_rel_error'].append(br)

        if tel > 0 and n_steps % tel != 0 and ep_acc != 0.0:
            self.history['episodic_reward'].append(float(ep_acc))
            self.history['episodic_cumulative_regret'].append(float(cum_reg))

        return self.M.copy()

    def get_indices(self):
        return (1 - self.gamma) * self.M.copy()

    def get_M(self):
        return self.M.copy()
