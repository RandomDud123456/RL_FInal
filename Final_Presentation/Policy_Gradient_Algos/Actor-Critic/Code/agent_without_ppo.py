"""
Ranking Policy Gradient Agent for RMAB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Architecture (from attached design doc):
  Actor  φ_θ(s_i) → score per arm
  Critic V_ψ(s_i) → value per arm state

Policy:
  Plackett-Luce top-K sampling for exploration + clean gradient

Loss:
  L = -L_actor(PPO) + λ_c * L_critic + λ_m * L_margin - λ_e * H(π)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ── Networks ─────────────────────────────────────────────────────────────────

class ActorNet(nn.Module):
    """
    Outputs a scalar score per arm given its state vector.
    Shared weights across arms (permutation-equivariant by design).
    """
    def __init__(self, state_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        states: [N, state_dim] - we are processing N arm states per batch, but each arm independently with shared weights.
        returns: scores [N]
        """
        return self.net(states).squeeze(-1)   # [N]


class CriticNet(nn.Module):
    """Estimates V(s_i) for each arm independently."""
    def __init__(self, state_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        states: [N, state_dim]
        returns: values [N]
        """
        return self.net(states).squeeze(-1)   # [N]


# ── Plackett-Luce sampling ────────────────────────────────────────────────────

def plackett_luce_top_k(scores: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample top-K arms without replacement proportional to exp(scores).
    Equivalent to Gumbel-max trick for PL sampling.

    Returns:
        selected: [K]  — indices of chosen arms
        log_prob: scalar — log P(A | scores) under PL
    """
    N = scores.shape[0]
    # Gumbel perturbation for PL sampling
    gumbels = -torch.empty_like(scores).exponential_().log()   # Gumbel(0,1)
    perturbed = scores + gumbels

    # top-K by perturbed scores
    selected = torch.topk(perturbed, K).indices   # [K]

    # Compute log P(A) under PL exactly
    log_prob = _pl_log_prob(scores, selected)

    return selected, log_prob


def _pl_log_prob(scores: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    """
    log P({a_1,...,a_K}) under Plackett-Luce given scores.
    Selected is the ordered selection sequence.

    Uses a large negative offset instead of boolean masking to avoid
    any inplace operations that break autograd.
    """
    N = scores.shape[0]
    log_p = torch.zeros(1, device=scores.device, dtype=scores.dtype).squeeze()
    # Work with a running "negated" mask added to scores — no inplace ops
    neg_inf_mask = torch.zeros(N, device=scores.device, dtype=scores.dtype)

    for k in range(len(selected)):
        arm = selected[k].item()
        masked_scores = scores + neg_inf_mask          # purely functional, no inplace
        denom = torch.logsumexp(masked_scores, dim=0)
        log_p = log_p + scores[arm] - denom
        # Build one-hot with scatter (no inplace on a grad-tracked tensor)
        one_hot = torch.zeros(N, device=scores.device, dtype=scores.dtype).scatter(
            0, torch.tensor([arm], device=scores.device), 1.0
        )
        neg_inf_mask = neg_inf_mask - 1e9 * one_hot   # new tensor each step

    return log_p


def pl_log_prob_batch(scores: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    """
    Vectorised log prob for a batch.
    scores:   [T, N]
    selected: [T, K]
    returns:  [T]
    """
    T = scores.shape[0]
    log_probs = torch.stack([
        _pl_log_prob(scores[t], selected[t]) for t in range(T)
    ])
    return log_probs


# ── Entropy of PL policy (approx via Monte-Carlo) ───────────────────────────

def pl_entropy_approx(scores: torch.Tensor, K: int, n_samples: int = 16) -> torch.Tensor:
    """Approximate H(π) via MC samples from PL."""
    log_probs = torch.stack([
        _pl_log_prob(scores, plackett_luce_top_k(scores, K)[0])
        for _ in range(n_samples)
    ])
    return -log_probs.mean()


# ── SoftSort (for auxiliary gradient flow) ────────────────────────────────────

def softsort(scores: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """
    Differentiable approximation of the permutation matrix.
    P[i,j] ≈ P(arm i is at rank j).
    scores: [N]
    returns: soft permutation [N, N]
    """
    N = scores.shape[0]
    sorted_scores, _ = torch.sort(scores, descending=True)
    # |scores_i - sorted_j| for all i, j
    diff = torch.abs(scores.unsqueeze(1) - sorted_scores.unsqueeze(0))   # [N, N]
    P = torch.softmax(-tau * diff, dim=1)
    return P


# ── Main Agent ────────────────────────────────────────────────────────────────

class RankingPGAgent:
    """
    Plackett-Luce + Advantage Actor-Critic for RMAB.

    Loss:
      L =
        - A_t log π(a|s)
        + λ_c * L_critic
        + λ_m * L_margin
        - λ_e * H(π)
    """

    def __init__(
        self,
        state_dim: int,
        N: int,
        M: int,
        hidden: int = 64,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        update_epochs: int = 4,
        lam_critic: float = 0.5,
        lam_margin: float = 0.1,
        lam_entropy: float = 0.01,
        margin_m: float = 0.5,
        tau_start: float = 1.0,
        tau_end: float = 10.0,
        tau_anneal_steps: int = 50_000,
        device: str = "cpu",
    ):
        self.N = N
        self.M = M
        self.gamma = gamma
        self.update_epochs = update_epochs

        self.lam_critic = lam_critic
        self.lam_margin = lam_margin
        self.lam_entropy = lam_entropy
        self.margin_m = margin_m

        self.tau_start = tau_start
        self.tau_end = tau_end
        self.tau_anneal_steps = tau_anneal_steps

        self.device = torch.device(device)

        self.actor = ActorNet(state_dim, hidden).to(self.device)
        self.critic = CriticNet(state_dim, hidden).to(self.device)

        self.opt_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self._reset_buffer()
        self.total_steps = 0

    # ─────────────────────────────────────────────────────────────
    # Buffer
    # ─────────────────────────────────────────────────────────────

    def _reset_buffer(self):
        self.buf = dict(
            states=[],
            next_states=[],
            selected=[],
            arm_rewards=[],
        )

    # ─────────────────────────────────────────────────────────────
    # τ schedule
    # ─────────────────────────────────────────────────────────────

    @property
    def tau(self) -> float:
        frac = min(self.total_steps / self.tau_anneal_steps, 1.0)
        return self.tau_start + frac * (self.tau_end - self.tau_start)

    # ─────────────────────────────────────────────────────────────
    # Action selection
    # ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def act(self, obs: np.ndarray):

        states_t = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device
        )

        scores = self.actor(states_t)

        selected, _ = plackett_luce_top_k(scores, self.M)

        return (
            selected.cpu().numpy(),
            scores.cpu().numpy()
        )

    # ─────────────────────────────────────────────────────────────
    # Store rollout transition
    # ─────────────────────────────────────────────────────────────

    def store(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        selected: np.ndarray,
        arm_rewards: np.ndarray,
    ):

        self.buf["states"].append(obs.copy())
        self.buf["next_states"].append(next_obs.copy())
        self.buf["selected"].append(selected.copy())
        self.buf["arm_rewards"].append(arm_rewards.copy())

        self.total_steps += 1

    # ─────────────────────────────────────────────────────────────
    # Update
    # ─────────────────────────────────────────────────────────────

    def update(self):

        T = len(self.buf["states"])

        if T == 0:
            return {}

        # ---------------------------------------------------------
        # Tensorise
        # ---------------------------------------------------------

        states_all = torch.tensor(
            np.stack(self.buf["states"]),
            dtype=torch.float32,
            device=self.device
        )  # [T, N, d]

        nstates_all = torch.tensor(
            np.stack(self.buf["next_states"]),
            dtype=torch.float32,
            device=self.device
        )

        selected_all = torch.tensor(
            np.stack(self.buf["selected"]),
            dtype=torch.long,
            device=self.device
        )  # [T, M]

        arm_rewards_all = torch.tensor(
            np.stack(self.buf["arm_rewards"]),
            dtype=torch.float32,
            device=self.device
        )  # [T, N]

        loss_info = {}

        # ---------------------------------------------------------
        # Multiple gradient epochs
        # ---------------------------------------------------------

        for _ in range(self.update_epochs):

            T, N, d = states_all.shape

            # -----------------------------------------------------
            # Critic forward
            # -----------------------------------------------------

            v_curr = self.critic(
                states_all.view(T * N, d)
            ).view(T, N)

            v_next = self.critic(
                nstates_all.view(T * N, d)
            ).view(T, N)

            # -----------------------------------------------------
            # TD advantage
            # -----------------------------------------------------

            adv_per_arm = (
                arm_rewards_all
                + self.gamma * v_next.detach()
                - v_curr.detach()
            )

            adv_t = torch.stack([
                adv_per_arm[t, selected_all[t]].mean()
                for t in range(T)
            ])

            # Advantage normalization
            adv_t = (
                (adv_t - adv_t.mean())
                / (adv_t.std() + 1e-8)
            )

            # -----------------------------------------------------
            # Actor forward
            # -----------------------------------------------------

            scores_all = self.actor(
                states_all.view(T * N, d)
            ).view(T, N)

            log_probs = pl_log_prob_batch(
                scores_all,
                selected_all
            )

            # -----------------------------------------------------
            # Standard policy gradient loss
            # -----------------------------------------------------

            L_actor = -(
                adv_t * log_probs
            ).mean()

            # -----------------------------------------------------
            # Critic loss
            # -----------------------------------------------------

            td_targets = (
                arm_rewards_all
                + self.gamma * v_next.detach()
            )

            L_critic = (
                (v_curr - td_targets) ** 2
            ).mean()

            # -----------------------------------------------------
            # Margin ranking loss
            # -----------------------------------------------------

            L_margin_total = torch.tensor(
                0.0,
                device=self.device
            )

            for t in range(T):

                s_scores = scores_all[t]

                pos_idx = selected_all[t]

                neg_mask = torch.ones(
                    N,
                    dtype=torch.bool,
                    device=self.device
                )

                neg_mask[pos_idx] = False

                neg_idx = neg_mask.nonzero(
                    as_tuple=True
                )[0]

                if len(neg_idx) == 0:
                    continue

                s_pos = s_scores[pos_idx].unsqueeze(1)
                s_neg = s_scores[neg_idx].unsqueeze(0)

                gaps = s_pos - s_neg

                L_margin_total += torch.clamp(
                    self.margin_m - gaps,
                    min=0
                ).mean()

            L_margin = L_margin_total / T

            # -----------------------------------------------------
            # Entropy bonus
            # -----------------------------------------------------

            mean_scores = scores_all.mean(dim=0)

            H = pl_entropy_approx(
                mean_scores,
                self.M,
                n_samples=8
            )

            # -----------------------------------------------------
            # Total loss
            # -----------------------------------------------------

            L_total = (
                L_actor
                + self.lam_critic * L_critic
                + self.lam_margin * L_margin
                - self.lam_entropy * H
            )

            # -----------------------------------------------------
            # Optim step
            # -----------------------------------------------------

            self.opt_actor.zero_grad()
            self.opt_critic.zero_grad()

            L_total.backward()

            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                0.5
            )

            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                0.5
            )

            self.opt_actor.step()
            self.opt_critic.step()

        loss_info = {
            "L_actor": L_actor.item(),
            "L_critic": L_critic.item(),
            "L_margin": L_margin.item(),
            "Entropy": H.item(),
            "L_total": L_total.item(),
        }

        self._reset_buffer()

        return loss_info

    # ─────────────────────────────────────────────────────────────
    # Greedy evaluation
    # ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def greedy_act(self, obs: np.ndarray):

        states_t = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device
        )

        scores = self.actor(states_t)

        return torch.topk(
            scores,
            self.M
        ).indices.cpu().numpy()