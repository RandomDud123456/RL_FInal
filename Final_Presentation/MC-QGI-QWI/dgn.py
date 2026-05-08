"""
DGN: Deep Gittins Network (tabular QGI target, neural function approximator).

Same Bellman structure as QGI on the **pulled arm’s** active transition:
  Q^x(s,1) ← r + γ max(Q^x(s',1), M(x))
  M(x)    ← M(x) + β (Q^x(x,1) − M(x))

Input: one-hot(s) ‖ one-hot(x). Output: scalar Q̂(s,x).
"""

import numpy as np
import time
import random
from collections import deque

from bellman_metrics import gittins_deep_bellman_rel_error
from metrics_common import greedy_whittle_arm, greedy_whittle_instant_reward

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class GittinsNet(nn.Module):
        """Maps (state, reference) one-hots → scalar Q(s, x)."""

        def __init__(self, N, hidden=(64, 128, 64)):
            super().__init__()
            d_in = 2 * N
            layers = []
            prev = d_in
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, s_oh, x_oh):
            return self.net(torch.cat([s_oh, x_oh], dim=-1))


class DGN:
    def __init__(self, env, gamma=0.9, lr=1e-3, tau=5e-3,
                 batch_size=32, buffer_size=10000, update_every=10,
                 beta_fn=None, epsilon=1.0, hidden=(64, 128, 64),
                 grad_clip_norm=1.0):
        if not TORCH_AVAILABLE:
            raise ImportError('PyTorch required for DGN')

        self.env = env
        self.K = env.K
        self.N = env.N
        self.gamma = gamma
        self.tau = float(tau)
        self.batch_size = batch_size
        self.update_every = update_every
        self.epsilon = epsilon
        self.grad_clip_norm = float(grad_clip_norm)

        self.beta_fn = beta_fn if beta_fn is not None else (lambda n: 0.05 if (n % 5 == 0) else 0.0)

        self.net = GittinsNet(self.N, hidden)
        self.target_net = GittinsNet(self.N, hidden)
        self.target_net.load_state_dict(self.net.state_dict())
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        self.M = np.zeros(self.N)
        self.buffer = deque(maxlen=buffer_size)
        self._eye = torch.eye(self.N)

        self.history = {
            'whittle_indices': [],
            'indices': [],
            'bre': [],
            'bellman_rel_error': [],
            'runtime': [],
            'avg_reward': [],
            'cumulative_reward': [],
            'cumulative_regret': [],
            'avg_reward_smooth': [],
            'policy_accuracy_pct': [],
            'q_delta_norm': [],
            'loss': [],
            'episodic_reward': [],
            'episodic_cumulative_regret': [],
        }
        self._rewards_since_log = []

    def _onehot(self, idx):
        return self._eye[int(idx)]

    def _predict(self, s, x, use_target=False):
        net = self.target_net if use_target else self.net
        with torch.no_grad():
            s_oh = self._onehot(s).unsqueeze(0)
            x_oh = self._onehot(x).unsqueeze(0)
            return float(net(s_oh, x_oh).squeeze().cpu().numpy())

    def _select_arm(self, states):
        """Same ranking as QGI: greedy on retirement level M."""
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.K))
        scores = [float(self.M[int(states[i])]) for i in range(self.K)]
        return int(np.argmax(scores))

    def _update_network(self):
        if len(self.buffer) < self.batch_size:
            return None
        batch = random.sample(self.buffer, self.batch_size)
        s_list = [b[0] for b in batch]
        r_list = [b[1] for b in batch]
        sn_list = [b[2] for b in batch]

        total = torch.tensor(0.0)
        for x in range(self.N):
            x_oh = self._onehot(x)
            s_oh = torch.stack([self._onehot(s) for s in s_list])
            sn_oh = torch.stack([self._onehot(s) for s in sn_list])
            x_b = x_oh.unsqueeze(0).expand(self.batch_size, -1)

            q_pred = self.net(s_oh, x_b).squeeze(-1)
            with torch.no_grad():
                q_sn = self.target_net(sn_oh, x_b).squeeze(-1)
            mx = float(self.M[x])
            target = torch.tensor(r_list, dtype=torch.float32, device=q_pred.device) + \
                self.gamma * torch.maximum(q_sn, torch.full_like(q_sn, mx))
            loss = nn.SmoothL1Loss()(q_pred, target)
            total = total + loss

        self.optimizer.zero_grad()
        total.backward()
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        return float(total.detach().cpu().item())

    def _soft_update(self):
        for p, tp in zip(self.net.parameters(), self.target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def _step_M(self, n):
        beta = self.beta_fn(n)
        if beta <= 0:
            return
        for x in range(self.N):
            q = self._predict(x, x, use_target=False)
            self.M[x] = self.M[x] + beta * (q - self.M[x])

    def train(self, n_steps=50000, true_indices=None, log_every=500, epsilon_decay=None,
              trace_episode_len=None):
        """
        true_indices: for ``bre``, compared to learned Gittins G=(1−γ)M.
        For regret / policy match, pass **Whittle** truth so the oracle matches other agents.
        """
        states = self.env.reset()
        start = time.time()
        cum_r = cum_reg = 0.0
        oracle_hits = 0
        ema_smooth = None
        ema_beta = 0.12
        prev_vec = None
        tel = int(trace_episode_len) if trace_episode_len else 0
        ep_acc = 0.0

        for n in range(1, n_steps + 1):
            if epsilon_decay is not None:
                self.epsilon = epsilon_decay(n, self.epsilon)

            arm = self._select_arm(states)
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

            s = int(states[arm])
            s_next = int(next_states[arm])
            self.buffer.append((s, float(reward), s_next))

            if n % self.update_every == 0:
                loss = self._update_network()
                self._soft_update()
                if loss is not None:
                    self.history['loss'].append(loss)

            self._step_M(n)
            states = next_states

            if tel > 0 and n % tel == 0:
                self.history['episodic_reward'].append(float(ep_acc))
                self.history['episodic_cumulative_regret'].append(float(cum_reg))
                ep_acc = 0.0

            if n % log_every == 0:
                self.history['runtime'].append(time.time() - start)
                self.history['indices'].append(self.M.copy())
                g = (1.0 - self.gamma) * self.M
                self.history['whittle_indices'].append(g.copy())

                if true_indices is not None:
                    t = np.asarray(true_indices, dtype=float)
                    bre = float(np.mean(np.abs(g - t)))
                    self.history['bre'].append(bre)

                if self._rewards_since_log:
                    wm = float(np.mean(self._rewards_since_log))
                    self.history['avg_reward'].append(wm)
                    ema_smooth = wm if ema_smooth is None else (1 - ema_beta) * ema_smooth + ema_beta * wm
                    self.history['avg_reward_smooth'].append(float(ema_smooth))
                    self._rewards_since_log = []

                self.history['cumulative_reward'].append(float(cum_r))
                if true_indices is not None:
                    self.history['cumulative_regret'].append(float(cum_reg))
                    self.history['policy_accuracy_pct'].append(100.0 * oracle_hits / float(log_every))
                    oracle_hits = 0
                else:
                    self.history['cumulative_regret'].append(float('nan'))
                    self.history['policy_accuracy_pct'].append(float('nan'))

                with torch.no_grad():
                    vec = torch.cat([p.flatten() for p in self.net.parameters()])
                v = vec.cpu().numpy()
                if prev_vec is None:
                    self.history['q_delta_norm'].append(0.0)
                else:
                    self.history['q_delta_norm'].append(float(np.linalg.norm(v - prev_vec)))
                prev_vec = v.copy()

                br = gittins_deep_bellman_rel_error(
                    self.env, self.gamma, self.M, self._predict)
                self.history['bellman_rel_error'].append(br)

        if tel > 0 and n_steps % tel != 0 and ep_acc != 0.0:
            self.history['episodic_reward'].append(float(ep_acc))
            self.history['episodic_cumulative_regret'].append(float(cum_reg))

        return self.M.copy()

    def get_indices(self):
        return (1.0 - self.gamma) * self.M.copy()
