"""
H-SRAC: Homogeneous State-Ranking Actor-Critic
================================================
A modular, problem-agnostic implementation of the H-SRAC algorithm
from "Strong Indexability and Policy Gradient Methods" (Vaddineni, 2026).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import warnings
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1.  Hyper-parameter container
# ---------------------------------------------------------------------------

@dataclass
class HSRACConfig:
    # Network
    actor_hidden: List[int] = field(default_factory=lambda: [16, 32])
    critic_hidden: List[int] = field(default_factory=lambda: [16, 32])

    # Training
    actor_lr:   float = 0.01
    critic_lr:  float = 0.001
    gamma:      float = 0.999
    num_episodes: int = 30000
    episode_len:  int = 3000

    # Budget enforcement (Newton solver)
    newton_iters: int = 50
    newton_tol:   float = 1e-6

    # Stochastic sensitivity  λ ~ U(lambda_lo, lambda_hi)
    lambda_lo:  float = 1.0
    lambda_hi:  float = 20.0

    # Evaluation & Misc
    seed: int = 42
    device: str = "cpu"   # Reverted to CPU as requested
    log_every: int = 50
    test_runs: int = 50   # Number of test runs for evaluation
    test_every: int = 50  # Interval (in episodes) to evaluate
    entropy_coef: float = 0.01


# ---------------------------------------------------------------------------
# 2.  Abstract environment interface
# ---------------------------------------------------------------------------

class RMABEnv(ABC):
    @property
    @abstractmethod
    def N(self) -> int:
        pass

    @property
    @abstractmethod
    def M(self) -> int:
        pass

    @property
    @abstractmethod
    def state_space(self) -> List:
        pass

    @property
    def state_dim(self) -> int:
        return len(self._encode_state(self.state_space[0]))

    @abstractmethod
    def reset(self) -> List:
        pass

    @abstractmethod
    def step(self, states: List, actions: List[int]) -> Tuple[List, List[float]]:
        pass

    @abstractmethod
    def _encode_state(self, state) -> np.ndarray:
        pass

    def encode_states(self, states: List) -> torch.Tensor:
        vecs = [self._encode_state(s) for s in states]
        return torch.tensor(np.stack(vecs), dtype=torch.float32)

    def encode_state_space(self) -> torch.Tensor:
        return self.encode_states(self.state_space)

    def state_to_index(self, state) -> int:
        return self.state_space.index(state)


# ---------------------------------------------------------------------------
# 3.  MLP helper
# ---------------------------------------------------------------------------

def build_mlp(in_dim: int, hidden: List[int], out_dim: int,
              activation=nn.LeakyReLU) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), activation()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# 4.  Actor  (Global Ranker)
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    def __init__(self, state_dim: int, hidden: List[int]):
        super().__init__()
        self.net = build_mlp(state_dim, hidden, 1)

    def forward(self, state_space_enc: torch.Tensor) -> torch.Tensor:
        return self.net(state_space_enc).squeeze(-1)


# ---------------------------------------------------------------------------
# 5.  Critic  (State-Specific Value Estimator)
# ---------------------------------------------------------------------------

class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden: List[int]):
        super().__init__()
        self.net = build_mlp(state_dim, hidden, 1)

    def forward(self, state_enc: torch.Tensor) -> torch.Tensor:
        return self.net(state_enc).squeeze(-1)


# ---------------------------------------------------------------------------
# 6.  Newton-Balanced Sigmoid  (Budget Regulator)
# ---------------------------------------------------------------------------

def newton_bias(logits: torch.Tensor, lam: float, M: int,
                n_iters: int = 50, tol: float = 1e-6) -> float:
    h = logits.detach().cpu().numpy().astype(np.float64)
    b = 0.0
    for _ in range(n_iters):
        z    = lam * h + b
        z    = np.clip(z, -500, 500)
        sig  = 1.0 / (1.0 + np.exp(-z))
        f    = sig.sum() - M
        df   = (sig * (1.0 - sig)).sum()
        if abs(df) < 1e-12:
            break
        b -= f / df
        if abs(f) < tol:
            break
    return float(b)

def selection_probs(logits: torch.Tensor, lam: float, b: float) -> torch.Tensor:
    return torch.sigmoid(lam * logits + b)


# ---------------------------------------------------------------------------
# 7.  H-SRAC Agent
# ---------------------------------------------------------------------------

class HSRAC:
    def __init__(self, env: RMABEnv, config: Optional[HSRACConfig] = None):
        self.env = env
        self.cfg = config or HSRACConfig()
        cfg = self.cfg

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.device = torch.device(cfg.device)

        sdim = env.state_dim
        self.actor  = Actor(sdim,  cfg.actor_hidden).to(self.device)
        self.critic = Critic(sdim, cfg.critic_hidden).to(self.device)

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=cfg.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self._ss_enc = env.encode_state_space().to(self.device)
        self.training_returns: List[float] = []

    def get_cached_policy(self):
        with torch.no_grad():
            cached_logits = self.actor(self._ss_enc)
        def policy(states):
            arm_logits = self._arm_logits(states, cached_logits)
            actions = [0] * self.env.N
            top_m = torch.topk(arm_logits, self.env.M).indices.tolist()
            for i in top_m:
                actions[i] = 1
            return actions
        return policy

    def train(self, num_episodes: Optional[int] = None,
              test_seeds: Optional[List[int]] = None,
              whittle_baseline: Optional[float] = None,
              env_cls=None, env_kwargs=None):
        cfg  = self.cfg
        n_ep = num_episodes or cfg.num_episodes
        env  = self.env
        test_history = []

        for ep in range(n_ep):
            states  = env.reset()
            ep_ret  = 0.0
            
            # λ-Annealing Curriculum
            progress = ep / n_ep
            current_lam_hi = cfg.lambda_lo + (cfg.lambda_hi - cfg.lambda_lo) * progress
            lam = float(np.random.uniform(cfg.lambda_lo, current_lam_hi))

            for _ in range(cfg.episode_len):
                logits     = self.actor(self._ss_enc)
                arm_logits = self._arm_logits(states, logits)

                b     = newton_bias(arm_logits, lam, env.M, cfg.newton_iters, cfg.newton_tol)
                probs = selection_probs(arm_logits, lam, b)
                actions = torch.bernoulli(probs.detach()).long().tolist()

                next_states, rewards = env.step(states, actions)
                ep_ret += sum(rewards)

                s_enc  = env.encode_states(states).to(self.device)
                sn_enc = env.encode_states(next_states).to(self.device)

                with torch.no_grad():
                    v_next = self.critic(sn_enc)

                v_curr    = self.critic(s_enc)
                r_tensor  = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                advantage = (r_tensor + cfg.gamma * v_next - v_curr).detach()
                
                # Advantage Normalization
                if advantage.shape[0] > 1:
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                td_target  = (r_tensor + cfg.gamma * v_next).detach()
                critic_loss = ((v_curr - td_target) ** 2).mean()

                self.critic_opt.zero_grad()
                critic_loss.backward()
                self.critic_opt.step()

                # Actor Update (with Entropy Regularization)
                a_tensor   = torch.tensor(actions, dtype=torch.float32, device=self.device)
                log_probs  = self._log_probs(arm_logits, lam, b, a_tensor)
                
                probs_ent = selection_probs(arm_logits, lam, b)
                entropy = -(probs_ent * torch.log(probs_ent + 1e-8) + 
                            (1 - probs_ent) * torch.log(1 - probs_ent + 1e-8)).mean()
                
                actor_loss = -(advantage * log_probs).mean() - cfg.entropy_coef * entropy

                self.actor_opt.zero_grad()
                actor_loss.backward()
                self.actor_opt.step()

                states = next_states

            self.training_returns.append(ep_ret)
            
            if (ep + 1) % cfg.log_every == 0:
                avg = np.mean(self.training_returns[-cfg.log_every:])
                print(f"Episode {ep+1:>6} | avg train return: {avg:+.1f}")
                
            if test_seeds is not None and (ep + 1) % cfg.test_every == 0:
                policy_fn = self.get_cached_policy()
                test_reward = evaluate_policy(env_cls, env_kwargs, policy_fn, test_seeds, cfg.episode_len, cfg.gamma)
                test_history.append((ep + 1, test_reward))
                print(f"   [Test] H-SRAC: {test_reward:.2f} | Whittle: {whittle_baseline if whittle_baseline else 'N/A'}")

        return test_history

    def _arm_logits(self, states: List, logits: torch.Tensor) -> torch.Tensor:
        indices = [self.env.state_to_index(s) for s in states]
        return logits[indices]

    @staticmethod
    def _log_probs(arm_logits: torch.Tensor, lam: float, b: float,
                   actions: torch.Tensor) -> torch.Tensor:
        logit_vals = lam * arm_logits + b
        log_p1 = -torch.nn.functional.softplus(-logit_vals)
        log_p0 = -torch.nn.functional.softplus( logit_vals)
        return actions * log_p1 + (1.0 - actions) * log_p0


# ===========================================================================
# Standardized Evaluation Logic
# ===========================================================================

def evaluate_policy(env_cls, env_kwargs, policy_fn, test_seeds, episode_len, gamma):
    total_rewards = []
    for seed in test_seeds:
        np.random.seed(seed)
        env = env_cls(**env_kwargs)
        states = env.reset()
        ep_ret = 0.0
        for t in range(episode_len):
            actions = policy_fn(states)
            next_states, rewards = env.step(states, actions)
            ep_ret += sum(rewards) * (gamma ** t)
            states = next_states
        total_rewards.append(ep_ret)
    return np.mean(total_rewards)


# ===========================================================================
# Analytical Deadline Whittle Index Formula
# ===========================================================================

def get_deadline_whittle_index(state, c, F_coef, gamma):
    D, B = state
    if B == 0: return 0.0
    elif 1 <= B <= D - 1: return 1.0 - c
    elif D <= B:
        F = lambda x: F_coef * (x**2)
        return (gamma ** (D - 1)) * F(B - D + 1) - (gamma ** (D - 1)) * F(B - D) + 1.0 - c
    return 0.0

def make_whittle_policy(env_N, env_M, c, F_coef, gamma):
    def policy(states):
        indices = [get_deadline_whittle_index(s, c, F_coef, gamma) for s in states]
        actions = [0] * env_N
        top_m = np.argsort(indices)[-env_M:]
        for i in top_m: actions[i] = 1
        return actions
    return policy

# ===========================================================================
# Deadline Scheduling Environment
# ===========================================================================

class DeadlineSchedulingEnv(RMABEnv):
    def __init__(self, N: int = 4, M: int = 1, D_max: int = 12, B_max: int = 9,
                 c: float = 0.5, F_coef: float = 0.2, arrival_prob: float = 0.7):
        self._N, self._M = N, M
        self.D_max, self.B_max = D_max, B_max
        self.c, self.F = c, lambda x: F_coef * (x ** 2)
        self.arrival_prob = arrival_prob
        self._state_space = [(0, 0)] + [(d, b) for d in range(1, D_max + 1) for b in range(0, B_max + 1)]

    @property
    def N(self): return self._N
    @property
    def M(self): return self._M
    @property
    def state_space(self): return self._state_space

    def reset(self):
        self._states = []
        for _ in range(self._N):
            if np.random.rand() < self.arrival_prob:
                self._states.append((np.random.randint(1, self.D_max + 1), np.random.randint(1, self.B_max + 1)))
            else: self._states.append((0, 0))
        return list(self._states)

    def step(self, states, actions):
        next_states, rewards = [], []
        for (D, B), a in zip(states, actions):
            if B > 0 and D > 1: r = (1 - self.c) * a
            elif B > 0 and D == 1: r = (1 - self.c) * a - self.F(max(B - a, 0))
            else: r = 0.0
            if D > 1: nD, nB = D - 1, max(B - a, 0)
            else:
                if np.random.rand() < self.arrival_prob:
                    nD, nB = np.random.randint(1, self.D_max + 1), np.random.randint(1, self.B_max + 1)
                else: nD, nB = 0, 0
            next_states.append((nD, nB)); rewards.append(r)
        return next_states, rewards

    def _encode_state(self, state):
        return np.array([state[0]/self.D_max, state[1]/self.B_max], dtype=np.float32)

    def state_to_index(self, state): return self._state_space.index(state)


if __name__ == "__main__":
    print("=" * 60)
    print("H-SRAC — Deadline Scheduling (Local Plotting)")
    print("=" * 60)

    experiments = [(100, 25)]
    all_results = []
    
    device = "cpu"
    print(f"Using device: {device}")

    for exp_N, exp_M in experiments:
        print(f"\nExperiment: N={exp_N}, M={exp_M}")
        env_kwargs = dict(N=exp_N, M=exp_M, D_max=12, B_max=9, c=0.5, arrival_prob=0.7)
        cfg = HSRACConfig(num_episodes=30000, test_every=50, log_every=50, device=device)
        
        np.random.seed(999) 
        test_seeds = [np.random.randint(0, 10000) for _ in range(cfg.test_runs)]
        
        whittle_policy = make_whittle_policy(exp_N, exp_M, 0.5, 0.2, cfg.gamma)
        whittle_avg = evaluate_policy(DeadlineSchedulingEnv, env_kwargs, whittle_policy, test_seeds, cfg.episode_len, cfg.gamma)
        print(f"Deadline Whittle Baseline: {whittle_avg:.2f}")

        agent = HSRAC(DeadlineSchedulingEnv(**env_kwargs), cfg)
        history = agent.train(test_seeds=test_seeds, whittle_baseline=whittle_avg, env_cls=DeadlineSchedulingEnv, env_kwargs=env_kwargs)
        
        # Generate Plot Immediately
        print(f"Generating plot for N={exp_N}, M={exp_M}...")
        plt.figure(figsize=(10, 6))
        eps, rewards = zip(*history)
        plt.plot(eps, rewards, label='H-SRAC (Test Reward)', marker='o', markersize=4)
        plt.axhline(y=whittle_avg, color='r', linestyle='--', label='Deadline Whittle Baseline')
        plt.title(f"Performance Comparison: N={exp_N}, M={exp_M}")
        plt.xlabel("Training Episodes")
        plt.ylabel("Total Discounted Reward")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(f"results_N{exp_N}_M{exp_M}.png")
        plt.close()
        print(f"Saved: results_N{exp_N}_M{exp_M}.png")
    
    print("\nAll experiments complete.")
