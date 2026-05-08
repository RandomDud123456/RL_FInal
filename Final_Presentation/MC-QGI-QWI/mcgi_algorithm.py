"""
MC-QWI / MC-QGI  —  v3: bug-fixed
===================================

BUGS FIXED from v2:
-------------------

BUG 1 (MC-QGI):
  (a) Blending off-policy Ĝ into max(Q,M) in the **one-step** Bellman backup biases
      the retirement fixed point.
  (b) Using V_mean (all transitions) in **ranking** dilutes indices with passive
      near-zero returns under concurrent control.
  FIX:
  (i) **TD + M-updates stay tabular QGI** (sound Bellman / retirement step).
  (ii) **Episode MC**: first-visit pull/passive returns with **infinite-horizon tail**
       γ^{T-t}·V̂(s_H) into G_mean / V_mean (for diagnostics / future use).
  (iii) **MC control in π**: when ``use_vhat_blend``, blend **pull-only** pooled
       Ĝ(s,1) (not V_mean) with the Gittins index for arm ranking — real MC signal
       in action selection without corrupting value iteration.

BUG 2 (MC-QWI MAE / subsidy):
  Earlier versions fed Q̃ into the TD bootstrap and/or tied the reported index to
  raw subsidy in inconsistent ways. Current design: **tabular QWI-style TD and
  subsidy** use **plain Q** only (same double-Q backup and diagonal Q-gap λ-update
  as ``qwi.QWI``). **Q̃(MC)** is used **only** for ``_compute_index`` / arm
  ranking (and optional ``use_vhat_blend`` extras), so MC corrects the index
  estimate without biasing the Bellman fixed-point iteration.

BUG 3 (QGI passive reward = 0, distorts Q):
  In rested bandits, passive arms are FROZEN — they don't transition and earn
  nothing. So r_eff=0 for passive IS correct in QGI mode. But since we skip
  the Q update for passive (a!=1), this never even executes. The issue was that
  the same r_eff=0 logic was incorrectly applied in QWI mode too (v2 fix was
  correct — passive in QWI earns r). Keeping that fix.

BUG 4 (V̂ never populated for QGI arms when passive):
  In episode_update, V_mean was only updated via mc.update(arm,s,a,G_t) where
  a is the recorded action. For QGI, passive steps (a=0) were skipped in
  Q-update but G_t WAS recorded in compute_mc_returns. However, the V_mean
  (action-agnostic) update needs to receive these returns. 
  FIX: MCValueEstimator.update() now always updates BOTH G_mean[arm,s,a]
  AND V_mean[arm,s] for every (arm,s,a,G_t) tuple — so passive-step returns
  populate V_mean correctly and Ṽ is properly informed.
"""

import numpy as np
from collections import defaultdict
from typing import Callable, Optional


# =============================================================================
# 1. MC Value Estimator
# =============================================================================

class MCValueEstimator:
    """
    Unified MC statistics store per arm.

    G_mean[arm, s, a] = running mean of G_t when action a was taken from s
                         → used for Q̃(s,a) in QWI TD bootstrap
    V_mean[arm, s]    = running mean of G_t from s regardless of action
                         → used for Ṽ(s) in QGI TD bootstrap + V̂ ranking blend

    FIX: Both are always updated together in update(), so QGI gets V_mean
    populated from ALL transitions (passive and active), not just pulls.
    """

    def __init__(self, n_arms: int, n_states: int, N0: float = 50.0):
        self.n_arms   = n_arms
        self.n_states = n_states
        self.N0       = float(N0)

        self.G_mean  = np.zeros((n_arms, n_states, 2))
        self.G_count = np.zeros((n_arms, n_states, 2))
        self.V_mean  = np.zeros((n_arms, n_states))
        self.V_count = np.zeros((n_arms, n_states))

    def update(self, arm: int, s: int, a: int, G_t: float):
        """
        Update G_mean[arm,s,a] AND V_mean[arm,s] from one observed return.
        Always updates both — this is critical for QGI to get V_mean from
        passive steps even though Q is not updated on passive steps.
        """
        n = self.G_count[arm, s, a]
        self.G_mean[arm, s, a]  += (G_t - self.G_mean[arm, s, a]) / (n + 1)
        self.G_count[arm, s, a] += 1

        m = self.V_count[arm, s]
        self.V_mean[arm, s]  += (G_t - self.V_mean[arm, s]) / (m + 1)
        self.V_count[arm, s] += 1

    def Q_tilde(self, arm: int, s: int, a: int, Q_val: float) -> float:
        """Q̃(s,a) = λ·G_mean(s,a) + (1-λ)·Q(s,a). Used for QWI."""
        n   = self.G_count[arm, s, a]
        lam = n / (n + self.N0)
        return lam * self.G_mean[arm, s, a] + (1.0 - lam) * Q_val

    def Q_tilde_both(self, arm: int, s: int, Q_row: np.ndarray) -> np.ndarray:
        """Q̃ for both actions. Q_row shape (2,)."""
        return np.array([
            self.Q_tilde(arm, s, 0, Q_row[0]),
            self.Q_tilde(arm, s, 1, Q_row[1]),
        ])

    def V_tilde(self, arm: int, s: int, Q_val: float) -> float:
        """
        Ṽ(s) = λ·V_mean(s) + (1-λ)·Q(s). Used for QGI.
        FIX: uses V_mean (action-agnostic) not G_mean[s,1] (pull-only).
        λ = V_count(s) / (V_count(s) + N0)
        """
        m   = self.V_count[arm, s]
        lam = m / (m + self.N0)
        return lam * self.V_mean[arm, s] + (1.0 - lam) * Q_val

    def ranking_score(self, arm: int, s: int, index_val: float,
                      C: float = 100.0, use_blend: bool = True) -> float:
        """
        Final ranking score = (1-c)·index + c·V̂(s).
        c = V_count(s) / (V_count(s) + C).
        """
        if not use_blend:
            return index_val
        m = self.V_count[arm, s]
        c = m / (m + C)
        return (1.0 - c) * index_val + c * self.V_mean[arm, s]

    def pooled_G(self, s: int, a: int) -> tuple[float, float]:
        """Weighted MC return mean and total visit count at (s,a) across arms."""
        num = 0.0
        den = 0.0
        for k in range(self.n_arms):
            c = float(self.G_count[k, s, a])
            if c > 0.0:
                num += float(self.G_mean[k, s, a]) * c
                den += c
        if den < 1e-12:
            return 0.0, 0.0
        return num / den, den

    def pooled_V(self, s: int) -> tuple[float, float]:
        """Weighted V_mean and total V_count at s across arms."""
        num = 0.0
        den = 0.0
        for k in range(self.n_arms):
            c = float(self.V_count[k, s])
            if c > 0.0:
                num += float(self.V_mean[k, s]) * c
                den += c
        if den < 1e-12:
            return 0.0, 0.0
        return num / den, den


# =============================================================================
# 2. First-visit MC Returns
# =============================================================================

def compute_mc_returns(trajectory: list, n_arms: int, gamma: float) -> dict:
    """
    First-visit MC: walk trajectory backwards, G = r + γ·G.
    Record G at first visit to each (arm, s, a).
    Returns: {(arm, s, a): [G_t values]}
    """
    returns = defaultdict(list)
    T = len(trajectory)
    for arm in range(n_arms):
        G       = 0.0
        visited = set()
        for t in reversed(range(T)):
            states_t, actions_t, rewards_t = trajectory[t][0], trajectory[t][1], trajectory[t][2]
            s = int(states_t[arm])
            a = int(actions_t[arm])
            r = float(rewards_t[arm])
            G = r + gamma * G
            if (s, a) not in visited:
                visited.add((s, a))
                returns[(arm, s, a)].append(G)
    return returns


def compute_first_visit_return_records(trajectory: list, n_arms: int, gamma: float) -> list:
    """
    Same backward first-visit MC as compute_mc_returns, but returns flat records
    ``(arm, s, a, t_step, G_partial)`` with G_partial = discounted return from
    step t to the **last reward** on that arm in the episode (no tail bootstrap).
    """
    records: list = []
    T = len(trajectory)
    for arm in range(n_arms):
        G = 0.0
        visited = set()
        for t in reversed(range(T)):
            states_t, actions_t, rewards_t = trajectory[t][0], trajectory[t][1], trajectory[t][2]
            s = int(states_t[arm])
            a = int(actions_t[arm])
            r = float(rewards_t[arm])
            G = r + gamma * G
            if (s, a) not in visited:
                visited.add((s, a))
                records.append((arm, s, a, int(t), float(G)))
    return records


# =============================================================================
# 3. Behavior Policy πβ
# =============================================================================

class BehaviorPolicy:
    """
    πβ ← (1-α_π)·πβ + α_π·π_greedy
    mix: 0=random, 1=greedy. Updated every env step.
    """

    def __init__(self, n_arms, budget, alpha_pi=1e-4,
                 mix_start=0.0, mix_end=1.0, freeze_mix=False):
        self.n_arms     = n_arms
        self.budget     = budget
        self.alpha_pi   = alpha_pi
        self.mix        = mix_start
        self.mix_end    = mix_end
        self.freeze_mix = freeze_mix

    def update(self):
        if not self.freeze_mix:
            self.mix = min(self.mix_end, self.mix + self.alpha_pi * (self.mix_end - self.mix))

    def select_actions(self, ranking: np.ndarray) -> np.ndarray:
        actions = np.zeros(self.n_arms, dtype=int)
        if np.random.rand() < (1.0 - self.mix):
            chosen = np.random.choice(self.n_arms, self.budget, replace=False)
        else:
            chosen = ranking[:self.budget]
        actions[chosen] = 1
        return actions


# =============================================================================
# 4. Core Agent
# =============================================================================

class MCIndexAgent:
    """
    MC-QWI / MC-QGI: v3 with all 4 bugs fixed.

    mode='qwi': Restless. Augmented double-Q like ``QWI``: TD bootstrap and μ(x)
                updates use **plain Q** only. Q̃ enters **only** in ``_compute_index``
                (MAE vs truth) and ``rank_arms`` (optional ``use_vhat_blend``).

    mode='qgi': Rested. Tabular QGI backups and M-updates unchanged; episode MC
                fills G_mean; optional ranking blends Ĝ(s,1) with (1-γ)M when
                use_vhat_blend. Index=(1-γ)M for metrics (Q̃ not in TD).
                Set ``match_qgi_exploration=True`` (+ ``epsilon_decay_fn``) to mirror tabular
                QGI's ε-greedy arm choice instead of πβ mixing.
    """

    def __init__(
        self,
        n_arms         : int,
        n_states       : int,
        budget         : int,
        mode           : str   = "qwi",
        gamma          : float = 0.99,
        alpha          : float = 0.05,
        N0             : float = 50.0,
        lr_lambda      : float = 0.001,
        alpha_pi       : float = 1e-4,
        conf_C         : float = 100.0,
        alpha_fn       : Optional[Callable[[int], float]] = None,
        subsidy_beta_fn: Optional[Callable[[int], float]] = None,
        use_mc_correction: bool = True,
        use_vhat_blend   : bool = True,
        anneal_behavior  : bool = True,
        q_clip           : Optional[float] = None,
        homogeneous      : bool = True,
        match_qgi_exploration: bool = False,
        epsilon          : float = 1.0,
        epsilon_decay_fn : Optional[Callable[[int, float], float]] = None,
    ):
        assert mode in ("qwi", "qgi")
        self.n_arms   = n_arms
        self.n_states = n_states
        self.budget   = budget
        self.mode     = mode
        self.gamma    = gamma
        # Identical arms: tabular QWI/QGI use *shared* Q / λ / M. Per-arm MC tables
        # split data — pool MC stats + symmetrize Q each step to match that inductive bias.
        self.homogeneous = bool(homogeneous)
        self.alpha    = alpha
        self.lr_lambda = lr_lambda
        self.alpha_fn  = alpha_fn
        self.subsidy_beta_fn = subsidy_beta_fn
        self.use_mc_correction = use_mc_correction
        self.use_vhat_blend    = use_vhat_blend
        self.conf_C            = conf_C

        span = 1.0 / max(1e-8, 1.0 - gamma)
        self._q_clip = float(q_clip) if q_clip is not None else span * 5.0 + 10.0

        # Augmented Q tables: shape (K, N, N, 2)
        self.Q  = np.random.uniform(0, 0.01, (n_arms, n_states, n_states, 2))
        self.QB = np.random.uniform(0, 0.01, (n_arms, n_states, n_states, 2)) if mode == "qwi" else None

        # MC value estimator (unified G_mean + V_mean)
        self.mc = MCValueEstimator(n_arms, n_states, N0=N0) if use_mc_correction else None

        # Subsidy / retirement
        if mode == "qwi":
            self.whittle_subsidy = np.zeros((n_arms, n_states))
            self.gittins_M       = None
        else:
            self.whittle_subsidy = None
            self.gittins_M       = np.zeros((n_arms, n_states))

        self.pi         = BehaviorPolicy(n_arms, budget, alpha_pi=alpha_pi, freeze_mix=not anneal_behavior)
        self.match_qgi_exploration = bool(match_qgi_exploration) and mode == "qgi"
        self.epsilon = float(epsilon)
        self._epsilon_decay_fn = epsilon_decay_fn
        self.step_count = 0
        self.history    = {}

    def _lr(self, n):
        return float(self.alpha_fn(int(n))) if self.alpha_fn else float(self.alpha)

    def _beta(self, n):
        return float(self.subsidy_beta_fn(int(n))) if self.subsidy_beta_fn else float(self.lr_lambda)

    def _bootstrap_tail_value_qgi(self, arm: int, s_tail: int) -> float:
        """Scalar tail V̂(s_H) ≈ max_x max(Q^x(s_H), M(x)) for MC return bootstrap."""
        assert self.gittins_M is not None
        st = int(s_tail)
        best = -1e18
        for x in range(self.n_states):
            v = max(float(self.Q[arm, x, st, 1]), float(self.gittins_M[arm, x]))
            best = max(best, v)
        return 0.0 if best < -1e17 else float(best)

    # --- Q̃ helpers ---

    def _Q_tilde_scalar(self, arm: int, s: int, a: int, Q_val: float) -> float:
        if self.mc is None:
            return float(Q_val)
        if self.homogeneous and self.n_arms > 1:
            g_hat, den = self.mc.pooled_G(s, a)
            lam = den / (den + self.mc.N0)
            return lam * g_hat + (1.0 - lam) * float(Q_val)
        return self.mc.Q_tilde(arm, s, a, Q_val)

    def _Qt_row(self, arm, s, Q_row):
        """Q̃ for both actions (QWI **index / ranking only**; not used in TD or μ updates)."""
        if self.mc is None:
            return np.asarray(Q_row, dtype=float).copy()
        if self.homogeneous and self.n_arms > 1:
            return np.array([
                self._Q_tilde_scalar(arm, s, 0, float(Q_row[0])),
                self._Q_tilde_scalar(arm, s, 1, float(Q_row[1])),
            ])
        return self.mc.Q_tilde_both(arm, s, np.asarray(Q_row, dtype=float))

    # --- QWI update ---

    def _update_Q_qwi(self, arm, s, a, r, s_next, n_step):
        """
        Augmented double-Q update for QWI (tabular: same structure as ``qwi.QWI``).
        base(x) = (1-a)·μ(x) + a·r
        TD target = base(x) + γ·Q_other(s', a*), with a* = argmax Q_pri(s',·).
        MC / Q̃ is **not** used here — only for ``_compute_index`` / ``rank_arms``.
        """
        assert self.QB is not None and self.whittle_subsidy is not None
        lr  = float(np.clip(self._lr(n_step), 1e-8, 1.0))
        a_i = int(a)

        update_A = np.random.rand() < 0.5
        Q_pri = self.Q  if update_A else self.QB
        Q_oth = self.QB if update_A else self.Q

        for x in range(self.n_states):
            mu_x = float(self.whittle_subsidy[arm, x])
            base = (1.0 - a_i) * mu_x + float(a_i) * float(r)

            q_next_pri = Q_pri[arm, x, s_next]
            a_star = int(np.argmax(q_next_pri))
            q_oth_astar = float(Q_oth[arm, x, s_next, a_star])

            td_target = base + self.gamma * q_oth_astar
            q_old = float(Q_pri[arm, x, s, a_i])
            Q_pri[arm, x, s, a_i] = np.clip((1.0 - lr) * q_old + lr * td_target,
                                             -self._q_clip, self._q_clip)

    def _subsidy_step_qwi(self, n_step):
        """μ(x) += β·diagonal Q-gap (tabular QWI); MC Q̃ is not used here."""
        beta = self._beta(n_step)
        if beta <= 0.0:
            return
        assert self.QB is not None and self.whittle_subsidy is not None
        for arm in range(self.n_arms):
            for x in range(self.n_states):
                qA = self.Q[arm, x, x]
                qB = self.QB[arm, x, x]
                gap = 0.5 * (float(qA[1] - qA[0]) + float(qB[1] - qB[0]))
                self.whittle_subsidy[arm, x] = np.clip(
                    self.whittle_subsidy[arm, x] + beta * gap, -20.0, 20.0)

    # --- QGI update ---

    def _update_Q_qgi(self, arm, s, a, r, s_next, n_step):
        """
        Pull-only backup for QGI. Skip passive steps.
        TD target = r + γ·max(Q^x(s'), M(x)) (tabular); n-step MC/TD at episode end.
        """
        assert self.gittins_M is not None
        if int(a) != 1:
            return
        lr  = float(np.clip(self._lr(n_step), 1e-8, 1.0))
        r_f = float(r)
        for x in range(self.n_states):
            q_next = float(self.Q[arm, x, int(s_next), 1])
            M_x = float(self.gittins_M[arm, x])
            td_target = r_f + self.gamma * max(q_next, M_x)
            q_old = float(self.Q[arm, x, s, 1])
            self.Q[arm, x, s, 1] = np.clip((1.0 - lr) * q_old + lr * td_target,
                                            -self._q_clip, self._q_clip)

    def _retirement_step_qgi(self, n_step):
        """M(x) += β·(Q^x(x,x) - M(x))."""
        beta = self._beta(n_step)
        if beta <= 0.0:
            return
        assert self.gittins_M is not None
        for arm in range(self.n_arms):
            for x in range(self.n_states):
                q_xx = float(self.Q[arm, x, x, 1])
                self.gittins_M[arm, x] = np.clip(
                    self.gittins_M[arm, x] + beta * (q_xx - self.gittins_M[arm, x]),
                    -100.0, 100.0)

    # --- Index computation: Q̃ gap for QWI, (1-γ)M for QGI ---

    def _compute_index(self, arm: int, s: int) -> float:
        """
        QWI: Q̃ diagonal gap at (x=s, s_phys=s) averaged over both nets.
             FIX: no longer returns raw whittle_subsidy (which diverged).
             The Q̃ gap IS the proper MC-corrected Whittle index estimate.

        QGI: (1-γ)·M(s).
        """
        if self.mode == "qwi":
            assert self.QB is not None
            Qt_A = self._Qt_row(arm, s, self.Q[arm, s, s])
            Qt_B = self._Qt_row(arm, s, self.QB[arm, s, s])
            return 0.5 * (float(Qt_A[1] - Qt_A[0]) + float(Qt_B[1] - Qt_B[0]))
        else:
            assert self.gittins_M is not None
            return float((1.0 - self.gamma) * self.gittins_M[arm, s])

    # --- STEP 4: Rank ---

    def _ranking_blend_V(self, arm: int, s: int, index_val: float) -> float:
        if self.mc is None or not self.use_vhat_blend:
            return index_val
        si = int(s)
        # QGI: use pull-only MC return Ĝ(s,1); V_mean mixes passive zeros and hurts π.
        if self.mode == "qgi":
            g_hat, den = self.mc.pooled_G(si, 1)
            c = float(den) / (float(den) + float(self.conf_C))
            # Cap MC influence: Ĝ is off-policy early; small blend nudges π without overriding λ.
            c = min(c, 0.08)
            return (1.0 - c) * index_val + c * float(g_hat)
        if self.homogeneous and self.n_arms > 1:
            v_hat, den = self.mc.pooled_V(si)
            c = den / (den + self.conf_C)
            return (1.0 - c) * index_val + c * v_hat
        return self.mc.ranking_score(
            arm, si, index_val, C=self.conf_C, use_blend=True
        )

    def rank_arms(self, states: np.ndarray) -> np.ndarray:
        scores = np.array([
            self._ranking_blend_V(arm, int(states[arm]), self._compute_index(arm, int(states[arm])))
            if self.mc is not None
            else self._compute_index(arm, int(states[arm]))
            for arm in range(self.n_arms)
        ])
        return np.argsort(scores)[::-1]

    # --- STEP 1: Select actions ---

    def select_actions(self, states: np.ndarray) -> np.ndarray:
        """
        QWI / default: softmax-style πβ mix toward greedy ranking.

        When ``match_qgi_exploration`` (QGI only): same rule as tabular ``QGI`` —
        decay ε each step, then with prob ε choose ``budget`` arms uniformly at random
        (without replacement), else greedy top-``budget`` by ``rank_arms``.
        """
        if self.match_qgi_exploration:
            n_step = int(self.step_count) + 1
            if self._epsilon_decay_fn is not None:
                self.epsilon = float(self._epsilon_decay_fn(n_step, float(self.epsilon)))
            ranking = self.rank_arms(states)
            actions = np.zeros(self.n_arms, dtype=int)
            B = min(int(self.budget), self.n_arms)
            if np.random.rand() < float(self.epsilon):
                chosen = np.random.choice(self.n_arms, size=B, replace=False)
            else:
                chosen = ranking[:B]
            actions[chosen] = 1
            return actions
        return self.pi.select_actions(self.rank_arms(states))

    # --- Online update (every step) ---

    def online_update(self, states, actions, rewards, next_states):
        n_step = self.step_count + 1
        for arm in range(self.n_arms):
            s, a, r, s2 = int(states[arm]), int(actions[arm]), float(rewards[arm]), int(next_states[arm])
            if self.mode == "qwi":
                self._update_Q_qwi(arm, s, a, r, s2, n_step)
            else:
                self._update_Q_qgi(arm, s, a, r, s2, n_step)
        if self.mode == "qwi":
            self._subsidy_step_qwi(n_step)
        else:
            self._retirement_step_qgi(n_step)
        if self.homogeneous and self.n_arms > 1:
            self._symmetrize_homogeneous_tables()
        self.step_count += 1
        if not self.match_qgi_exploration:
            self.pi.update()

    # --- End-of-episode MC update ---

    def episode_update(self, trajectory: list):
        """
        First-visit MC returns → MCValueEstimator. QGI uses infinite-horizon tail
        bootstrap on each sample when ``next_states`` are stored on the trajectory.
        """
        if not self.use_mc_correction or self.mc is None:
            return
        if not trajectory:
            return
        T = len(trajectory)
        last = trajectory[-1]
        has_next = len(last) >= 4
        final_s = None
        if has_next:
            final_s = np.asarray(last[3], dtype=int)
        records = compute_first_visit_return_records(trajectory, self.n_arms, self.gamma)
        for arm, s, a, t, G_part in records:
            if has_next and self.mode == "qgi" and self.gittins_M is not None:
                s_h = int(final_s[arm])
                n_tail = T - int(t)
                tail_v = self._bootstrap_tail_value_qgi(arm, s_h)
                g_use = float(G_part) + (self.gamma ** n_tail) * tail_v
            else:
                g_use = float(G_part)
            self.mc.update(arm, int(s), int(a), g_use)

    def _symmetrize_homogeneous_tables(self) -> None:
        """Match tabular QWI/QGI: identical arms share one value function."""
        if self.mode == "qwi" and self.QB is not None and self.whittle_subsidy is not None:
            self.Q[:] = np.mean(self.Q, axis=0, keepdims=True)
            self.QB[:] = np.mean(self.QB, axis=0, keepdims=True)
            mu = np.mean(self.whittle_subsidy, axis=0, keepdims=True)
            self.whittle_subsidy[:] = mu
        elif self.mode == "qgi" and self.gittins_M is not None:
            self.Q[:] = np.mean(self.Q, axis=0, keepdims=True)
            mu_m = np.mean(self.gittins_M, axis=0, keepdims=True)
            self.gittins_M[:] = mu_m

    # --- Diagnostics ---

    def get_all_indices(self, states: np.ndarray) -> np.ndarray:
        return np.array([self._compute_index(arm, int(states[arm])) for arm in range(self.n_arms)])

    def get_indices(self, hetero: bool = False) -> np.ndarray:
        if hetero:
            out = np.zeros((self.n_arms, self.n_states))
            for k in range(self.n_arms):
                for s in range(self.n_states):
                    out[k, s] = self._compute_index(k, s)
            return out
        out = np.zeros(self.n_states)
        if self.homogeneous and self.n_arms > 1:
            for s in range(self.n_states):
                out[s] = float(self._compute_index(0, s))
        else:
            for s in range(self.n_states):
                out[s] = float(np.mean([self._compute_index(k, s) for k in range(self.n_arms)]))
        return out

    def diagnostics(self, states: np.ndarray) -> dict:
        if self.mc is not None:
            lam = self.mc.G_count / (self.mc.G_count + self.mc.N0)
            mean_lam = float(lam.mean())
        else:
            mean_lam = 0.0
        mix = float(1.0 - self.epsilon) if self.match_qgi_exploration else float(self.pi.mix)
        d = {"mean_lambda": mean_lam, "behavior_mix": mix,
             "mean_index": float(self.get_all_indices(states).mean())}
        if self.mode == "qwi" and self.whittle_subsidy is not None:
            d["whittle_mean"] = float(self.whittle_subsidy.mean())
        if self.mode == "qgi" and self.gittins_M is not None:
            d["gittins_M_mean"] = float(self.gittins_M.mean())
        return d


# =============================================================================
# 5. Training Loop
# =============================================================================

def train(env, agent: MCIndexAgent, n_episodes: int, episode_len: int,
          verbose: bool = True, log_every: int = 100) -> dict:
    logs = {"rewards": [], "ep_totals": [], "mean_index": [], "behavior_mix": [], "mean_lambda": []}
    for ep in range(n_episodes):
        states, trajectory, ep_reward = env.reset(), [], 0.0
        for _ in range(episode_len):
            actions                    = agent.select_actions(states)
            next_states, rewards, done = env.step(actions)
            trajectory.append(
                (states.copy(), actions.copy(), rewards.copy(), next_states.copy())
            )
            agent.online_update(states, actions, rewards, next_states)
            ep_reward += float(rewards.sum())
            states = next_states
            if done: break
        agent.episode_update(trajectory)
        d = agent.diagnostics(states)
        logs["rewards"].append(ep_reward / episode_len)
        logs["ep_totals"].append(ep_reward)
        logs["mean_index"].append(d["mean_index"])
        logs["behavior_mix"].append(d["behavior_mix"])
        logs["mean_lambda"].append(d["mean_lambda"])
        if verbose and ep % log_every == 0:
            print(f"Ep {ep:4d}/{n_episodes} | Reward: {ep_reward/episode_len:7.4f} | "
                  f"Index: {d['mean_index']:6.3f} | πβ: {d['behavior_mix']:.3f} | λ: {d['mean_lambda']:.3f}")
    return logs


def mc_ablation_kwargs(ablation: str) -> dict:
    """
    Map ``compare_rma_four`` ``--mc_ablation`` flags to ``MCIndexAgent`` keyword args.

    ``full`` keeps defaults (MC correction on, V̂ ranking blend on, πβ mix annealing on).
    """
    a = (ablation or "full").strip().lower()
    if a in ("full", ""):
        return {}
    if a == "no_mc":
        return {"use_mc_correction": False}
    if a == "no_vhat":
        return {"use_vhat_blend": False}
    if a == "no_anneal":
        return {"anneal_behavior": False}
    raise ValueError(
        f"unknown mc_ablation {ablation!r}; expected full|no_mc|no_vhat|no_anneal"
    )


def train_rmab(env, agent, total_steps, episode_len, log_every,
               true_whittle, true_gittins, hetero, verbose=False):
    """Step-based training for comparison with tabular benchmarks."""
    from metrics_common import greedy_whittle_concurrent_reward
    mode         = agent.mode
    oracle_truth = np.asarray(true_whittle if mode == "qwi" else true_gittins, dtype=float)
    agent.history = {"bre": [], "bre_gittins": [], "whittle_indices": [],
                     "avg_reward": [], "cumulative_reward": [], "cumulative_regret": [],
                     "steps_logged": [], "episodic_reward": [], "episodic_cumulative_regret": []}
    step_count, cum_r, cum_reg, window = 0, 0.0, 0.0, []
    while step_count < total_steps:
        states, trajectory = env.reset(), []
        ep_r = 0.0
        for _ in range(episode_len):
            if step_count >= total_steps: break
            r_star  = greedy_whittle_concurrent_reward(env, oracle_truth, states, agent.budget)
            actions = agent.select_actions(states)
            next_states, rewards, done = env.step_concurrent(actions)
            trajectory.append(
                (states.copy(), actions.copy(), rewards.copy(), next_states.copy())
            )
            agent.online_update(states, actions, rewards, next_states)
            r_step = float(np.sum(rewards))
            ep_r += r_step
            cum_r += r_step; cum_reg += r_star - r_step; window.append(r_step)
            step_count += 1
            if step_count % log_every == 0:
                snap = agent.get_indices(hetero=hetero)
                err  = float(np.mean(np.abs(snap - oracle_truth)))
                agent.history["whittle_indices"].append(snap.copy())
                key = "bre" if mode == "qwi" else "bre_gittins"
                agent.history[key].append(err)
                agent.history["steps_logged"].append(step_count)
                agent.history["avg_reward"].append(float(np.mean(window)))
                agent.history["cumulative_reward"].append(cum_r)
                agent.history["cumulative_regret"].append(cum_reg)
                window.clear()
                if verbose:
                    tag = "Whittle" if mode == "qwi" else "Gittins"
                    mix_disp = (
                        float(1.0 - agent.epsilon)
                        if getattr(agent, "match_qgi_exploration", False)
                        else float(agent.pi.mix)
                    )
                    print(f"MC-{mode.upper()} step {step_count}/{total_steps} | MAE {tag}: {err:.4f} | mix={mix_disp:.3f}")
            states = next_states
            if done: break
        agent.history["episodic_reward"].append(float(ep_r))
        agent.history["episodic_cumulative_regret"].append(float(cum_reg))
        agent.episode_update(trajectory)
    return agent.history


# =============================================================================
# 6. Baseline QWI
# =============================================================================

class BaselineQWI:
    def __init__(self, n_arms, n_states, budget, gamma=0.99, alpha=0.05,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=50000, lr_lambda=0.01):
        self.n_arms=n_arms; self.n_states=n_states; self.budget=budget
        self.gamma=gamma; self.alpha=alpha
        self.eps_start=epsilon_start; self.eps_end=epsilon_end; self.eps_decay=epsilon_decay
        self.lr_lambda=lr_lambda
        self.Q=np.zeros((n_arms,n_states,2)); self.subsidy=np.zeros(n_arms); self.steps=0

    @property
    def epsilon(self):
        return self.eps_end + (self.eps_start-self.eps_end)*np.exp(-self.steps/self.eps_decay)

    def select_actions(self, states):
        actions=np.zeros(self.n_arms,dtype=int)
        if np.random.rand()<self.epsilon:
            chosen=np.random.choice(self.n_arms,self.budget,replace=False)
        else:
            idx=np.array([self.Q[a,int(states[a]),1]-self.Q[a,int(states[a]),0] for a in range(self.n_arms)])
            chosen=np.argsort(idx)[-self.budget:]
        actions[chosen]=1; return actions

    def online_update(self, states, actions, rewards, next_states):
        for arm in range(self.n_arms):
            s,a=int(states[arm]),int(actions[arm]); r,s2=float(rewards[arm]),int(next_states[arm])
            r_eff=(r-self.subsidy[arm]) if a==1 else r
            td=r_eff+self.gamma*np.max(self.Q[arm,s2])
            self.Q[arm,s,a]+=self.alpha*(td-self.Q[arm,s,a])
            self.subsidy[arm]=np.clip(self.subsidy[arm]+self.lr_lambda*(self.Q[arm,s,1]-self.Q[arm,s,0]),-20,20)
        self.steps+=1

    def episode_update(self, trajectory): pass


# =============================================================================
# 7. Synthetic RMAB Environment
# =============================================================================

class SyntheticRMAB:
    def __init__(self, n_arms, n_states, budget, seed=42, restless=True):
        self.n_arms=n_arms; self.n_states=n_states; self.budget=budget; self.restless=restless
        rng=np.random.RandomState(seed)
        self.P=np.zeros((n_arms,n_states,2,n_states)); self.R=np.zeros((n_arms,n_states,2))
        for arm in range(n_arms):
            for a in range(2):
                for s in range(n_states):
                    av=rng.uniform(0.5,2.0,n_states)
                    av[min(s+1,n_states-1)]+=(2.0 if a==1 else 0)
                    av[max(s-1,0)]+=(0 if a==1 else 1.5)
                    self.P[arm,s,a]=rng.dirichlet(av)
            for s in range(n_states):
                self.R[arm,s,0]=s/(n_states-1)*0.5
                self.R[arm,s,1]=s/(n_states-1)+rng.uniform(0,0.1)
        self.states=np.zeros(n_arms,dtype=int)

    def reset(self):
        self.states=np.random.randint(0,self.n_states,size=self.n_arms); return self.states.copy()

    def step(self, actions):
        rewards=np.zeros(self.n_arms); next_states=np.zeros(self.n_arms,dtype=int)
        for arm in range(self.n_arms):
            s,a=self.states[arm],int(actions[arm]); rewards[arm]=self.R[arm,s,a]
            if self.restless or a==1:
                next_states[arm]=np.random.choice(self.n_states,p=self.P[arm,s,a])
            else:
                next_states[arm]=s
        self.states=next_states; return next_states.copy(),rewards,False

    def step_concurrent(self, actions):
        return self.step(actions)


# =============================================================================
# 8. Comparison
# =============================================================================

def run_comparison(n_arms=10, n_states=5, budget=3, n_episodes=1000, episode_len=50, n_seeds=3, mode="qwi"):
    restless=(mode=="qwi")
    print("="*65)
    print(f"  MC-{mode.upper()} v3 (fixed) | Arms={n_arms} States={n_states} Budget={budget}")
    print(f"  {'Restless' if restless else 'Rested'} | Eps={n_episodes} EpLen={episode_len} Seeds={n_seeds}")
    print("="*65)
    results={f"MC-{mode.upper()}":[],"Baseline-QWI":[]}
    for seed in range(n_seeds):
        print(f"\n--- Seed {seed} ---")
        np.random.seed(seed)
        eps_decay=n_episodes*episode_len//4
        env_mc=SyntheticRMAB(n_arms,n_states,budget,seed=seed,restless=restless)
        env_bl=SyntheticRMAB(n_arms,n_states,budget,seed=seed,restless=restless)
        agent_mc=MCIndexAgent(
            n_arms=n_arms,
            n_states=n_states,
            budget=budget,
            mode=mode,
            gamma=0.99,
            alpha=0.05,
            N0=50.0,
            lr_lambda=0.001,
            alpha_pi=1e-4,
            conf_C=100.0,
            homogeneous=True,
        )
        agent_bl=BaselineQWI(n_arms=n_arms,n_states=n_states,budget=budget,
                             gamma=0.99,alpha=0.05,epsilon_start=1.0,epsilon_end=0.05,
                             epsilon_decay=eps_decay,lr_lambda=0.01)
        print(f"Training MC-{mode.upper()}...")
        logs_mc=train(env_mc,agent_mc,n_episodes,episode_len,verbose=True,log_every=200)
        print("Training Baseline QWI...")
        logs_bl=[]
        s_=env_bl.reset()
        for ep in range(n_episodes):
            s_=env_bl.reset(); ep_r=0.0; traj=[]
            for _ in range(episode_len):
                act=agent_bl.select_actions(s_); ns_,rw_,dn_=env_bl.step(act)
                traj.append((s_.copy(),act.copy(),rw_.copy()))
                agent_bl.online_update(s_,act,rw_,ns_); ep_r+=float(rw_.sum()); s_=ns_
                if dn_: break
            logs_bl.append(ep_r/episode_len)
            if ep%200==0: print(f"  [Baseline] Ep {ep:4d} | Reward: {ep_r/episode_len:.4f} | ε: {agent_bl.epsilon:.3f}")
        results[f"MC-{mode.upper()}"].append(logs_mc["rewards"])
        results["Baseline-QWI"].append(logs_bl)
    print("\n"+"="*65+"  RESULTS  "+"="*65)
    for name,curves in results.items():
        arr=np.array(curves); last_k=min(100,arr.shape[1])
        fm=arr[:,-last_k:].mean(); fs=arr[:,-last_k:].std()
        thresh=0.8*fm; conv=[]
        for c in curves:
            sm=np.convolve(c,np.ones(20)/20,mode="valid")
            above=np.where(sm>thresh)[0]; conv.append(int(above[0]) if len(above) else n_episodes)
        print(f"\n{name}: Final reward {fm:.4f}±{fs:.4f} | Convergence ep {np.mean(conv):.0f}")
    return results


if __name__ == "__main__":
    run_comparison(n_arms=10,n_states=5,budget=3,n_episodes=1000,episode_len=50,n_seeds=3,mode="qwi")
    run_comparison(n_arms=10,n_states=5,budget=3,n_episodes=1000,episode_len=50,n_seeds=3,mode="qgi")
