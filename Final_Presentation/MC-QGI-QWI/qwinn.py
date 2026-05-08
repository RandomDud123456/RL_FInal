"""
QWINN: Deep Q-Network for Whittle Index  (Robledo et al. 2022)
--------------------------------------------------------------
Deep RL version of QWI for RMAB.
Neural network replaces Q-table. Stores experience tuples for
BOTH active and passive actions.

Architecture:
  Input  : [current_state_s, reference_state_x]  (2 scalars or embeddings)
  Output : [Q(s,x,0), Q(s,x,1)]  (passive and active Q-values)
  Hidden : 3 layers (64, 128, 64) + ReLU

Loss:
  MSE between Q_target and Q_theta over minibatch × all reference states x

Target:
  active  : r(s) + γ max_v Q^x_θ'(s', v)
  passive : λ(x) + γ max_v Q^x_θ'(s', v)
"""

import numpy as np
import time
from collections import deque
import random

from bellman_metrics import whittle_deep_bellman_rel_error
from metrics_common import greedy_whittle_arm, greedy_whittle_instant_reward

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not found. Install with: pip install torch")


# ------------------------------------------------------------------
# Neural Network
# ------------------------------------------------------------------

class WhittleNet(nn.Module):
    """
    Maps (state, reference_state) → [Q(s,x,0), Q(s,x,1)]
    """
    def __init__(self, N, hidden=(64, 128, 64)):
        super().__init__()
        # Input: one-hot state (N) + one-hot reference state (N) = 2N
        input_dim = 2 * N
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 2))  # Q(s,x,0) and Q(s,x,1)
        self.net = nn.Sequential(*layers)

    def forward(self, s_onehot, x_onehot):
        inp = torch.cat([s_onehot, x_onehot], dim=-1)
        return self.net(inp)   # shape (batch, 2)


# ------------------------------------------------------------------
# QWINN Agent
# ------------------------------------------------------------------

class QWINN:
    """
    Deep Q-Network for Whittle Index on RMAB.

    Parameters
    ----------
    env          : RMABEnvironment
    gamma        : discount factor
    lr           : neural network learning rate
    tau          : soft update parameter for target network
    batch_size   : minibatch size
    buffer_size  : replay buffer capacity
    update_every : how often to do network update (κ in paper)
    beta_fn      : callable(n) -> slow timescale learning rate for λ
    epsilon      : exploration rate
    hidden       : hidden layer sizes
    grad_clip_norm : max norm for backprop (0 to disable)
    lam_clip     : clip λ after each update (None = auto from R_max)
    """

    def __init__(self, env, gamma=0.9, lr=5e-3, tau=1e-3,
                 batch_size=32, buffer_size=10000, update_every=10,
                 beta_fn=None, epsilon=1.0, hidden=(64, 128, 64),
                 grad_clip_norm=1.0, lam_clip=None):

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for QWINN")

        self.env          = env
        self.K            = env.K
        self.N            = env.N
        self.gamma        = gamma
        self.tau          = tau
        self.batch_size   = batch_size
        self.update_every = update_every
        self.epsilon      = epsilon
        self.grad_clip_norm = float(grad_clip_norm)
        r_max = float(np.max(env.R))
        r_min = float(np.min(env.R))
        scale = r_max / max(1e-8, 1.0 - gamma)
        span = (r_max - r_min) / max(1e-8, 1.0 - gamma)
        self._lam_clip = float(lam_clip) if lam_clip is not None else max(scale, span) + 15.0

        if beta_fn is None:
            self.beta_fn = lambda n: 0.2 if (n % 10 == 0) else 0.0
        else:
            self.beta_fn = beta_fn

        # Networks
        self.net        = WhittleNet(self.N, hidden)
        self.target_net = WhittleNet(self.N, hidden)
        self.target_net.load_state_dict(self.net.state_dict())
        self.optimizer  = optim.Adam(self.net.parameters(), lr=lr)

        # Replay buffer stores: (s, a, r, s_next)
        # QWINN stores tuples for BOTH active AND passive actions
        self.buffer = deque(maxlen=buffer_size)

        # Whittle index estimates λ[x]
        self.lam = np.zeros(self.N)

        # One-hot encoding helpers
        self._eye = torch.eye(self.N)

        # Logging
        self.history = {
            'whittle_indices': [],
            'loss': [],
            'bre': [],
            'bellman_rel_error': [],
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

    # ------------------------------------------------------------------
    # Helper: one-hot encoding
    # ------------------------------------------------------------------

    def _onehot(self, idx):
        """Return one-hot tensor for state index."""
        return self._eye[idx]

    def _q_values(self, s, x, use_target=False):
        """Get Q(s, x, :) from network. Returns numpy (2,)."""
        net = self.target_net if use_target else self.net
        with torch.no_grad():
            s_oh = self._onehot(s).unsqueeze(0)   # (1, N)
            x_oh = self._onehot(x).unsqueeze(0)   # (1, N)
            q = net(s_oh, x_oh)                    # (1, 2)
        return q.squeeze(0).numpy()

    # ------------------------------------------------------------------
    # Arm selection
    # ------------------------------------------------------------------

    def _select_arm(self, states):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.K)
        else:
            indices = [self.lam[states[i]] for i in range(self.K)]
            return np.argmax(indices)

    # ------------------------------------------------------------------
    # Network update
    # ------------------------------------------------------------------

    def _update_network(self):
        """Sample minibatch and update network via MSE loss."""
        if len(self.buffer) < self.batch_size:
            return None

        batch = random.sample(self.buffer, self.batch_size)
        # Each tuple: (s, a, r, s_next)
        s_list      = [b[0] for b in batch]
        a_list      = [b[1] for b in batch]
        r_list      = [b[2] for b in batch]
        s_next_list = [b[3] for b in batch]

        total_loss = torch.tensor(0.0)

        for x in range(self.N):
            x_oh = self._onehot(x)
            lam_x = float(self.lam[x])

            # Build batch tensors
            s_oh      = torch.stack([self._onehot(s) for s in s_list])       # (B, N)
            s_next_oh = torch.stack([self._onehot(s) for s in s_next_list])  # (B, N)
            x_oh_b    = x_oh.unsqueeze(0).expand(self.batch_size, -1)        # (B, N)

            # Predicted Q values (primary network)
            q_pred = self.net(s_oh, x_oh_b)   # (B, 2)

            # Target Q values (target network)
            with torch.no_grad():
                q_next = self.target_net(s_next_oh, x_oh_b)   # (B, 2)
                q_next_max = q_next.max(dim=1).values           # (B,)

            # Compute targets per sample
            targets = torch.zeros(self.batch_size, 2)
            for k, (a, r) in enumerate(zip(a_list, r_list)):
                if a == 1:
                    # Active: real reward
                    targets[k, 1] = r + self.gamma * q_next_max[k]
                    targets[k, 0] = q_pred[k, 0].detach()  # don't update passive
                else:
                    # Passive: synthetic reward = λ(x)
                    targets[k, 0] = lam_x + self.gamma * q_next_max[k]
                    targets[k, 1] = q_pred[k, 1].detach()  # don't update active

            loss = nn.SmoothL1Loss()(q_pred, targets)
            total_loss = total_loss + loss

        self.optimizer.zero_grad()
        total_loss.backward()
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        return total_loss.item()

    def _soft_update(self):
        """θ' ← τθ' + (1-τ)θ"""
        for p, tp in zip(self.net.parameters(), self.target_net.parameters()):
            tp.data.copy_(self.tau * tp.data + (1 - self.tau) * p.data)

    def _update_lambda(self, n):
        """Slow timescale update for Whittle indices."""
        beta = self.beta_fn(n)
        if beta > 0:
            for x in range(self.N):
                q = self._q_values(x, x, use_target=False)
                self.lam[x] = self.lam[x] + beta * (q[1] - q[0])
                self.lam[x] = float(np.clip(self.lam[x], -self._lam_clip, self._lam_clip))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, n_steps=50000, true_indices=None, log_every=500,
              epsilon_decay=None, trace_episode_len=None):
        """
        Run QWINN training.

        Parameters
        ----------
        n_steps      : total steps
        true_indices : ground truth Whittle indices for BRE
        log_every    : logging frequency
        epsilon_decay: callable(n, eps) -> new eps
        """
        states = self.env.reset()
        start  = time.time()
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

            # Store ACTIVE tuple
            self.buffer.append((states[arm], 1, reward, next_states[arm]))

            # Store PASSIVE tuples for ALL non-pulled arms  ← key QWINN feature
            for i in range(self.K):
                if i != arm:
                    self.buffer.append((states[i], 0, 0.0, next_states[i]))

            states = next_states

            # Network update every κ steps
            if n % self.update_every == 0:
                loss = self._update_network()
                self._soft_update()
                if loss is not None:
                    self.history['loss'].append(loss)

            # Slow timescale λ update
            self._update_lambda(n)

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
                    self.history['avg_reward'].append(win_mean)
                    ema_smooth = (
                        win_mean if ema_smooth is None
                        else (1 - ema_beta) * ema_smooth + ema_beta * win_mean
                    )
                    self.history['avg_reward_smooth'].append(float(ema_smooth))
                    self._rewards_since_log = []

                self.history['cumulative_reward'].append(float(cum_r))
                if true_indices is not None:
                    self.history['cumulative_regret'].append(float(cum_reg))
                    self.history['policy_accuracy_pct'].append(
                        100.0 * float(oracle_hits) / float(log_every))
                    oracle_hits = 0
                else:
                    self.history['cumulative_regret'].append(float('nan'))
                    self.history['policy_accuracy_pct'].append(float('nan'))

                with torch.no_grad():
                    vec = torch.cat([p.detach().flatten() for p in self.net.parameters()])
                v = vec.cpu().numpy()
                if q_prev is None:
                    self.history['q_delta_norm'].append(0.0)
                else:
                    self.history['q_delta_norm'].append(float(np.linalg.norm(v - q_prev)))
                q_prev = v.copy()

                if true_indices is not None:
                    bre = np.mean(np.abs(self.lam - true_indices))
                    self.history['bre'].append(bre)

                br = whittle_deep_bellman_rel_error(
                    self.env, self.gamma, self.lam,
                    lambda s, x: self._q_values(s, x, use_target=False))
                self.history['bellman_rel_error'].append(br)

        if tel > 0 and n_steps % tel != 0 and ep_acc != 0.0:
            self.history['episodic_reward'].append(float(ep_acc))
            self.history['episodic_cumulative_regret'].append(float(cum_reg))

        return self.lam.copy()

    def get_indices(self):
        return self.lam.copy()
