"""
Machine Repair RMAB
═══════════════════
N machines, M can be repaired per round.

State:  s ∈ {0, 1, …, S-1}   (0 = perfect, S-1 = broken)
Reward: r(s) = (S-1) - s      (perfect machine gives S-1, broken gives 0)

Transitions:
  Active  (repair):  s → max(s-1, 0)  with prob p_fix,
                     s → s            with prob 1-p_fix
  Passive (run):     s → min(s+1,S-1) with prob p_deg,
                     s → s            with prob 1-p_deg

Whittle index — computed EXACTLY via value iteration on the
ν-subsidized single-arm MDP for each state.  No approximation.

Policies compared:
  1. Whittle index   (exact, optimal reference)
  2. Ranking-PG      (our learned policy, from agent.py)
  3. Greedy-state    (always repair worst machines first, i.e. highest s)
  4. Random
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from copy import deepcopy

# ─────────────────────────────────────────────────────────────────────────────
# Problem parameters  (small so Whittle is tractable)
# ─────────────────────────────────────────────────────────────────────────────
S      = 6        # states per arm:  0..5
N      = 8        # number of arms
M      = 3        # arms activated per round
GAMMA  = 0.95     # discount
P_FIX  = 0.9      # prob of improving one level when active
P_DEG  = 0.4      # prob of degrading one level when passive
REWARD = lambda s: float(S - 1 - s)   # r(s) = 5,4,3,2,1,0

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class MachineRepairEnv:
    """
    Vectorised multi-arm environment.
    obs: [N]  integer states,  also returns normalised [N,1] for the NN.
    """
    def __init__(self, N=N, M=M, S=S, p_fix=P_FIX, p_deg=P_DEG, seed=0):
        self.N, self.M, self.S = N, M, S
        self.p_fix, self.p_deg = p_fix, p_deg
        self.rng = np.random.default_rng(seed)
        self.states = None
        self.reset()

    def reset(self):
        self.states = self.rng.integers(0, self.S, size=self.N)
        return self._obs()

    def _obs(self):
        # Normalised float obs for NN:  [N, 1]
        return (self.states / (self.S - 1)).astype(np.float32).reshape(self.N, 1)

    def step(self, action_idx: np.ndarray):
        """action_idx: [M] indices of arms to repair."""
        active = np.zeros(self.N, dtype=bool)
        active[action_idx] = True

        new_states = self.states.copy()
        rewards    = np.zeros(self.N)

        for i in range(self.N):
            s = self.states[i]
            if active[i]:
                # Repair: improve with prob p_fix
                if self.rng.random() < self.p_fix:
                    new_states[i] = max(s - 1, 0)
            else:
                # Run: degrade with prob p_deg
                if self.rng.random() < self.p_deg:
                    new_states[i] = min(s + 1, self.S - 1)
            rewards[i] = REWARD(new_states[i])

        self.states = new_states
        return self._obs(), rewards.sum(), rewards

    def state_raw(self):
        return self.states.copy()


# ─────────────────────────────────────────────────────────────────────────────
# Exact Whittle index via value iteration on subsidized single-arm MDP
# ─────────────────────────────────────────────────────────────────────────────

def build_single_arm_transitions(S, p_fix, p_deg):
    """
    Returns P_active[s,s'], P_passive[s,s'] transition matrices.
    """
    Pa = np.zeros((S, S))
    Pp = np.zeros((S, S))
    for s in range(S):
        # Active
        if s > 0:
            Pa[s, s-1] = p_fix
            Pa[s, s]   = 1 - p_fix
        else:
            Pa[s, s] = 1.0
        # Passive
        if s < S-1:
            Pp[s, s+1] = p_deg
            Pp[s, s]   = 1 - p_deg
        else:
            Pp[s, s] = 1.0
    return Pa, Pp


def whittle_index_vi(S, p_fix, p_deg, gamma, reward_fn,
                     nu_lo=-20.0, nu_hi=20.0, tol=1e-3, vi_tol=1e-6):
    """
    Compute Whittle index W(s) for each state s via bisection.

    For each state s, W(s) is the subsidy ν at which the optimal policy
    for the ν-subsidized MDP is indifferent between active/passive at state s.

    Bisection over ν; for each ν, solve the subsidized MDP via value iteration.
    """
    Pa, Pp = build_single_arm_transitions(S, p_fix, p_deg)
    r = np.array([reward_fn(s) for s in range(S)], dtype=float)

    def solve_subsidized_mdp(nu):
        """
        Subsidized reward: r(s,passive) = r(s) + nu, r(s,active) = r(s)
        Returns V[s], pi[s] (pi=1 → active, pi=0 → passive).
        """
        V = np.zeros(S)
        for _ in range(10_000):
            Q_act  = r + gamma * Pa @ V
            Q_pas  = (r + nu) + gamma * Pp @ V
            V_new  = np.maximum(Q_act, Q_pas)
            if np.max(np.abs(V_new - V)) < vi_tol:
                break
            V = V_new
        pi = (Q_act >= Q_pas).astype(int)   # 1=active preferred
        return V, pi, Q_act, Q_pas

    whittle = np.zeros(S)
    for s in range(S):
        lo, hi = nu_lo, nu_hi
        for _ in range(60):   # bisection iterations
            mid = (lo + hi) / 2
            _, pi, Qa, Qp = solve_subsidized_mdp(mid)
            # At W(s): Q_act(s) == Q_pas(s)  →  indifferent
            # If Qa[s] > Qp[s]: active preferred → subsidy too low → increase nu
            if Qa[s] >= Qp[s]:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        whittle[s] = (lo + hi) / 2

    return whittle


# Pre-compute once
WHITTLE = whittle_index_vi(S, P_FIX, P_DEG, GAMMA, REWARD)
print(f"Whittle indices per state: { {s: round(WHITTLE[s],3) for s in range(S)} }")


# ─────────────────────────────────────────────────────────────────────────────
# Policies
# ─────────────────────────────────────────────────────────────────────────────

class WhittlePolicy:
    """Rank arms by their Whittle index W(s_i), activate top-M."""
    def __init__(self, whittle: np.ndarray, M: int):
        self.whittle = whittle
        self.M = M

    def act(self, obs_raw: np.ndarray) -> np.ndarray:
        scores = self.whittle[obs_raw]           # [N]
        return np.argsort(-scores)[:self.M]

    def greedy_act(self, obs: np.ndarray, env) -> np.ndarray:
        return self.act(env.state_raw())


class GreedyStatePolicy:
    """Always repair the M machines with highest (worst) state."""
    def __init__(self, M: int):
        self.M = M

    def greedy_act(self, obs: np.ndarray, env) -> np.ndarray:
        return np.argsort(-env.state_raw())[:self.M]


class RandomPolicy:
    def __init__(self, N: int, M: int, seed: int = 99):
        self.N, self.M = N, M
        self.rng = np.random.default_rng(seed)

    def greedy_act(self, obs: np.ndarray, env=None) -> np.ndarray:
        return self.rng.choice(self.N, size=self.M, replace=False)


# ─────────────────────────────────────────────────────────────────────────────
# Ranking-PG Agent  (copy of agent.py, self-contained here for portability)
# ─────────────────────────────────────────────────────────────────────────────

class ActorNet(nn.Module):
    def __init__(self, state_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CriticNet(nn.Module):
    def __init__(self, state_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _pl_log_prob(scores: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    N = scores.shape[0]
    log_p = scores.new_zeros(())
    neg_inf_mask = scores.new_zeros(N)
    for k in range(len(selected)):
        arm = selected[k].item()
        denom = torch.logsumexp(scores + neg_inf_mask, dim=0)
        log_p = log_p + scores[arm] - denom
        one_hot = scores.new_zeros(N).scatter(
            0, torch.tensor([arm], device=scores.device), 1.0)
        neg_inf_mask = neg_inf_mask - 1e9 * one_hot
    return log_p


def pl_log_prob_batch(scores, selected):
    return torch.stack([_pl_log_prob(scores[t], selected[t])
                        for t in range(scores.shape[0])])


def plackett_luce_top_k(scores, K):
    gumbels  = -torch.empty_like(scores).exponential_().log()
    selected = torch.topk(scores + gumbels, K).indices
    return selected, _pl_log_prob(scores, selected)


def pl_entropy_approx(scores, K, n=8):
    lps = torch.stack([_pl_log_prob(scores, plackett_luce_top_k(scores, K)[0])
                       for _ in range(n)])
    return -lps.mean()


class RankingPGAgent:
    def __init__(self, state_dim, N, M,
                 hidden=64, lr_actor=3e-4, lr_critic=1e-3,
                 gamma=0.99, eps_clip=0.2, ppo_epochs=4,
                 lam_critic=0.5, lam_margin=0.1, lam_entropy=0.01,
                 margin_m=0.5, tau_start=1.0, tau_end=10.0,
                 tau_anneal_steps=50_000, device="cpu"):
        self.N, self.M = N, M
        self.gamma, self.eps_clip, self.ppo_epochs = gamma, eps_clip, ppo_epochs
        self.lam_critic, self.lam_margin, self.lam_entropy = lam_critic, lam_margin, lam_entropy
        self.margin_m = margin_m
        self.tau_start, self.tau_end = tau_start, tau_end
        self.tau_anneal_steps = tau_anneal_steps
        self.device = torch.device(device)
        self.actor  = ActorNet(state_dim, hidden).to(self.device)
        self.critic = CriticNet(state_dim, hidden).to(self.device)
        self.opt_a  = optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.opt_c  = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self._reset_buf()
        self.total_steps = 0

    def _reset_buf(self):
        self.buf = dict(states=[], next_states=[], selected=[], log_probs=[], arm_rewards=[])

    @property
    def tau(self):
        return self.tau_start + min(self.total_steps / self.tau_anneal_steps, 1.0) * (self.tau_end - self.tau_start)

    @torch.no_grad()
    def act(self, obs):
        s = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        sc = self.actor(s)
        sel, lp = plackett_luce_top_k(sc, self.M)
        return sel.cpu().numpy(), lp.item(), sc.cpu().numpy()

    @torch.no_grad()
    def greedy_act(self, obs, env=None):
        s = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        return torch.topk(self.actor(s), self.M).indices.cpu().numpy()

    def store(self, obs, next_obs, sel, lp, arm_r):
        self.buf["states"].append(obs.copy())
        self.buf["next_states"].append(next_obs.copy())
        self.buf["selected"].append(sel.copy())
        self.buf["log_probs"].append(lp)
        self.buf["arm_rewards"].append(arm_r.copy())
        self.total_steps += 1

    def update(self):
        T = len(self.buf["states"])
        if T == 0: return {}
        dev = self.device
        SA  = torch.tensor(np.stack(self.buf["states"]),      dtype=torch.float32, device=dev)
        NSA = torch.tensor(np.stack(self.buf["next_states"]), dtype=torch.float32, device=dev)
        SEL = torch.tensor(np.stack(self.buf["selected"]),    dtype=torch.long,    device=dev)
        OLP = torch.tensor(self.buf["log_probs"],             dtype=torch.float32, device=dev)
        AR  = torch.tensor(np.stack(self.buf["arm_rewards"]), dtype=torch.float32, device=dev)

        for _ in range(self.ppo_epochs):
            T_, N, d = SA.shape
            vc = self.critic(SA.view(T_*N, d)).view(T_, N)
            vn = self.critic(NSA.view(T_*N, d)).view(T_, N)
            adv_arm = AR + self.gamma * vn.detach() - vc.detach()
            adv_t   = torch.stack([adv_arm[t, SEL[t]].mean() for t in range(T_)])

            sc_all = self.actor(SA.view(T_*N, d)).view(T_, N)
            nlp    = pl_log_prob_batch(sc_all, SEL)
            rho    = torch.exp(nlp - OLP)
            L_act  = -torch.min(rho * adv_t,
                                torch.clamp(rho, 1-self.eps_clip, 1+self.eps_clip) * adv_t).mean()

            td_tgt = AR + self.gamma * vn.detach()
            L_cri  = ((vc - td_tgt)**2).mean()

            L_mar = torch.tensor(0.0, device=dev)
            for t in range(T_):
                pos = SEL[t]
                neg_mask = torch.ones(N, dtype=torch.bool, device=dev)
                neg_mask[pos] = False
                neg = neg_mask.nonzero(as_tuple=True)[0]
                if len(neg) == 0: continue
                gaps = sc_all[t][pos].unsqueeze(1) - sc_all[t][neg].unsqueeze(0)
                L_mar = L_mar + torch.clamp(self.margin_m - gaps, min=0).mean()
            L_mar = L_mar / T_

            H = pl_entropy_approx(sc_all.mean(0), self.M)
            L = L_act + self.lam_critic*L_cri + self.lam_margin*L_mar - self.lam_entropy*H

            self.opt_a.zero_grad(); self.opt_c.zero_grad()
            L.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(),  0.5)
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.opt_a.step(); self.opt_c.step()

        self._reset_buf()
        return {"L_total": L.item(), "L_actor": L_act.item(),
                "L_critic": L_cri.item(), "adv": adv_t.mean().item()}


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def rollout(env: MachineRepairEnv, policy, horizon: int, gamma: float,
            agent=None, store=False) -> float:
    obs = env.reset()
    total, disc = 0.0, 1.0
    for _ in range(horizon):
        if store:
            action, lp, _ = agent.act(obs)
        else:
            action = policy.greedy_act(obs, env)
            lp = 0.0
        next_obs, reward, arm_rewards = env.step(action)
        if store:
            agent.store(obs, next_obs, action, lp, arm_rewards)
        total += disc * reward
        disc  *= gamma
        obs    = next_obs
    return total


def evaluate(policy, N, M, S, p_fix, p_deg, horizon, gamma, n_eps=20,
             is_agent=False, seed_offset=5000) -> tuple:
    returns = []
    for ep in range(n_eps):
        env = MachineRepairEnv(N=N, M=M, S=S, p_fix=p_fix, p_deg=p_deg,
                               seed=seed_offset + ep)
        r = rollout(env, policy, horizon, gamma, agent=policy if is_agent else None,
                    store=False)
        returns.append(r)
    a = np.array(returns)
    return a.mean(), a.std()


def train(
    N=N, M=M, S=S, p_fix=P_FIX, p_deg=P_DEG,
    gamma=GAMMA,
    n_episodes=800,
    horizon=200,
    update_freq=5,
    eval_every=50,
    n_eval=20,
    seed=42,
    verbose=True,
):
    train_env = MachineRepairEnv(N=N, M=M, S=S, p_fix=p_fix, p_deg=p_deg, seed=seed)
    agent = RankingPGAgent(
        state_dim=1, N=N, M=M, hidden=64,
        lr_actor=3e-4, lr_critic=1e-3,
        gamma=gamma, eps_clip=0.2, ppo_epochs=4,
        lam_critic=0.5, lam_margin=0.1, lam_entropy=0.01,
        margin_m=0.5, tau_start=1.0, tau_end=10.0,
        tau_anneal_steps=n_episodes * horizon // 2,
    )

    whittle_pol  = WhittlePolicy(WHITTLE, M)
    greedy_pol   = GreedyStatePolicy(M)
    random_pol   = RandomPolicy(N, M, seed=seed)

    hist = {k: [] for k in ["ep", "agent", "agent_std",
                             "whittle", "whittle_std",
                             "greedy", "greedy_std",
                             "random", "random_std"]}

    for ep in range(1, n_episodes + 1):
        ep_env = MachineRepairEnv(N=N, M=M, S=S, p_fix=p_fix, p_deg=p_deg, seed=seed + ep)
        rollout(ep_env, None, horizon, gamma, agent=agent, store=True)

        if ep % update_freq == 0:
            agent.update()

        if ep % eval_every == 0:
            am, as_ = evaluate(agent,       N,M,S,p_fix,p_deg,horizon,gamma,n_eval,is_agent=True)
            wm, ws  = evaluate(whittle_pol, N,M,S,p_fix,p_deg,horizon,gamma,n_eval,is_agent=False)
            gm, gs  = evaluate(greedy_pol,  N,M,S,p_fix,p_deg,horizon,gamma,n_eval,is_agent=False)
            rm, rs  = evaluate(random_pol,  N,M,S,p_fix,p_deg,horizon,gamma,n_eval,is_agent=False)

            hist["ep"].append(ep)
            hist["agent"].append(am);   hist["agent_std"].append(as_)
            hist["whittle"].append(wm); hist["whittle_std"].append(ws)
            hist["greedy"].append(gm);  hist["greedy_std"].append(gs)
            hist["random"].append(rm);  hist["random_std"].append(rs)

            if verbose:
                gap = 100*(wm - am)/max(abs(wm), 1e-6)
                print(f"[Ep {ep:4d}] Agent:{am:6.1f}±{as_:.1f}  "
                      f"Whittle:{wm:6.1f}±{ws:.1f}  "
                      f"Greedy:{gm:6.1f}  Random:{rm:6.1f}  "
                      f"Gap-to-Whittle:{gap:.1f}%  τ={agent.tau:.1f}")

    return hist, agent


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot(hist, whittle_indices, save="machine_repair_results.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")

    # ── Left: learning curves ────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#161b22")
    clr = {"agent":   "#58a6ff",
           "whittle": "#ffa657",
           "greedy":  "#3fb950",
           "random":  "#f85149"}

    for key, label, ls in [
        ("whittle", "Whittle index (optimal ref)", "--"),
        ("agent",   "Ranking-PG (ours)",           "-"),
        ("greedy",  "Greedy-state",                ":"),
        ("random",  "Random",                      "-."),
    ]:
        m  = np.array(hist[key])
        s  = np.array(hist[key + "_std"])
        ep = hist["ep"]
        ax.plot(ep, m, label=label, color=clr[key], lw=2, ls=ls)
        ax.fill_between(ep, m-s, m+s, color=clr[key], alpha=0.12)

    ax.set_xlabel("Training Episodes", color="#8b949e", fontsize=11)
    ax.set_ylabel("Discounted Return",  color="#8b949e", fontsize=11)
    ax.set_title(f"Machine Repair RMAB  N={N}, M={M}, S={S}",
                 color="#e6edf3", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#8b949e")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    ax.legend(framealpha=0.2, labelcolor="#e6edf3",
              facecolor="#161b22", edgecolor="#30363d", fontsize=9)

    # ── Right: Whittle index profile ─────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#161b22")
    states = np.arange(S)
    bars = ax2.bar(states, whittle_indices, color=clr["whittle"], alpha=0.8, width=0.6)
    ax2.axhline(0, color="#8b949e", lw=0.8, ls="--")

    # Annotate bars
    for bar, v in zip(bars, whittle_indices):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.05,
                 f"{v:.2f}", ha="center", va="bottom", color="#e6edf3", fontsize=9)

    ax2.set_xlabel("Machine State s  (0=perfect, S-1=broken)",
                   color="#8b949e", fontsize=11)
    ax2.set_ylabel("Whittle Index W(s)", color="#8b949e", fontsize=11)
    ax2.set_title("Whittle Index Profile\n(Higher = more urgent to repair)",
                  color="#e6edf3", fontsize=12, fontweight="bold")
    ax2.set_xticks(states)
    ax2.tick_params(colors="#8b949e")
    for sp in ax2.spines.values(): sp.set_edgecolor("#30363d")

    # Annotate intuition
    ax2.annotate("← Passive (run) preferred",
                 xy=(0, whittle_indices[0]),
                 xytext=(0.3, whittle_indices[0] - 0.3),
                 color="#8b949e", fontsize=8)
    ax2.annotate("Active (repair) preferred →",
                 xy=(S-1, whittle_indices[S-1]),
                 xytext=(S-2.5, whittle_indices[S-1] + 0.2),
                 color="#8b949e", fontsize=8)

    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Plot saved → {save}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Verify Whittle index intuition
# ─────────────────────────────────────────────────────────────────────────────

def verify_whittle(whittle: np.ndarray):
    """
    Sanity checks:
    1. W(s) should be monotonically increasing in s
       (worse machines have higher priority to repair).
    2. W(0) should be negative (no benefit to repairing a perfect machine).
    3. W(S-1) should be positive (broken machine must be repaired).
    """
    print("\n── Whittle Index Verification ──")
    print(f"  Values: {np.round(whittle, 3)}")
    mono = all(whittle[i] <= whittle[i+1] for i in range(len(whittle)-1))
    print(f"  Monotone increasing (worse → higher priority): {mono} {'✓' if mono else '✗'}")
    print(f"  W(0)={whittle[0]:.3f} < 0 (perfect machine, low priority): "
          f"{'✓' if whittle[0] < 0 else '✗'}")
    print(f"  W(S-1)={whittle[-1]:.3f} > 0 (broken machine, high priority): "
          f"{'✓' if whittle[-1] > 0 else '✗'}")

    # Cross-check: at subsidy = W(s), Q_active(s) ≈ Q_passive(s)
    Pa, Pp = build_single_arm_transitions(S, P_FIX, P_DEG)
    r = np.array([REWARD(s) for s in range(S)], dtype=float)
    print("\n  Cross-check  Q_act(s) ≈ Q_pas(s) at ν = W(s):")
    for s in range(S):
        nu = whittle[s]
        V = np.zeros(S)
        for _ in range(5000):
            Qa = r + GAMMA * Pa @ V
            Qp = (r + nu) + GAMMA * Pp @ V
            V_new = np.maximum(Qa, Qp)
            if np.max(np.abs(V_new - V)) < 1e-8: break
            V = V_new
        diff = abs(Qa[s] - Qp[s])
        print(f"    s={s}: W={nu:.3f}  |Q_act - Q_pas| = {diff:.2e}  {'✓' if diff < 1e-3 else '✗'}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    verify_whittle(WHITTLE)

    print(f"\nTraining Ranking-PG on Machine Repair RMAB  (N={N}, M={M}, S={S})")
    hist, agent = train(
        N=N, M=M, S=S, p_fix=P_FIX, p_deg=P_DEG,
        gamma=GAMMA,
        n_episodes=800,
        horizon=200,
        update_freq=5,
        eval_every=50,
        n_eval=20,
        verbose=True,
    )

    plot(hist, WHITTLE, save="machine_repair_results.png")

    # Final gap-to-Whittle summary
    final_agent   = hist["agent"][-1]
    final_whittle = hist["whittle"][-1]
    final_greedy  = hist["greedy"][-1]
    print(f"\n── Final Results ──")
    print(f"  Whittle (optimal ref):  {final_whittle:.2f}")
    print(f"  Ranking-PG (ours):      {final_agent:.2f}  "
          f"(gap: {100*(final_whittle - final_agent)/max(abs(final_whittle),1e-6):.1f}%)")
    print(f"  Greedy-state:           {final_greedy:.2f}  "
          f"(gap: {100*(final_whittle - final_greedy)/max(abs(final_whittle),1e-6):.1f}%)")
    print(f"  Random:                 {hist['random'][-1]:.2f}")