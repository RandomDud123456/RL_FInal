"""
Training & Evaluation — Ranking PG vs Baselines
Deadline Scheduling (NeurWIN Section 5.2 exact setup)

Run:
  python train.py --N 10 --M 1 --episodes 2000
"""

import argparse, time
import numpy as np
from agent import RankingPGAgent
BACKEND = "torch"

from deadline_env import DeadlineSchedulingEnv
from baselines import RandomPolicy, GreedyDeadlinePolicy, GreedyRatioPolicy, WhittleIndexPolicy

DISCOUNT   = 0.99
HORIZON    = 300
UPDATE_FREQ = 5
EVAL_EVERY = 50
EVAL_EPS   = 10
SEED       = 42
STATE_DIM  = 2


def rollout(env, policy, horizon, use_env_in_act=False,
            agent=None, store=False, gamma=DISCOUNT):
    obs = env.reset()
    total, disc = 0.0, 1.0
    for _ in range(horizon):
        if use_env_in_act:
            action = policy.act(obs, env=env)
            lp = 0.0
        elif store:
            action, lp, _ = agent.act(obs)
        else:
            action = policy.greedy_act(obs)
            lp = 0.0
        next_obs, reward, arm_rewards = env.step(action)
        if store and agent is not None:
            agent.store(obs, next_obs, action, lp, arm_rewards)
        total += disc * reward
        disc  *= gamma
        obs    = next_obs
    return total


def evaluate(env_proto, agent_or_policy, n_eps, horizon, use_env=False):
    returns = []
    for ep in range(n_eps):
        e = DeadlineSchedulingEnv(env_proto.N, env_proto.M, seed=10_000 + ep)
        r = rollout(e, agent_or_policy, horizon, use_env_in_act=use_env)
        returns.append(r)
    a = np.array(returns)
    return a.mean(), a.std()


def train(N, M, n_episodes=2000, verbose=True):
    env = DeadlineSchedulingEnv(N, M, seed=SEED)

    kw = dict(state_dim=STATE_DIM, N=N, M=M, hidden=64,
              lr_actor=3e-4, lr_critic=1e-3, gamma=DISCOUNT,
              lam_margin=0.1, lam_entropy=0.01, margin_m=0.5,
              tau_start=1.0, tau_end=10.0,
              tau_anneal_steps=n_episodes * HORIZON // 2)
    if BACKEND == "torch":
        kw.update(eps_clip=0.2, ppo_epochs=4, device="cpu")
    else:
        kw["seed"] = SEED

    agent     = RankingPGAgent(**kw)
    rand_pol  = RandomPolicy(N, M, seed=SEED)
    dead_pol  = GreedyDeadlinePolicy(N, M)
    ratio_pol = GreedyRatioPolicy(N, M)
    whittle_pol = WhittleIndexPolicy(
        N,
        M,
        c0=0.0,
        beta=0.99,
        penalty_coeff=1.0
    )
    # ev = DeadlineSchedulingEnv(N, M, seed=SEED)
    # wm, ws = evaluate(ev, whittle_pol, EVAL_EPS, HORIZON, use_env=True)
    # print(f"Whittle Index Policy  |  Mean: {wm:.1f}  Std: {ws:.1f}\n")
    hist = {k: [] for k in ["episodes",
                              "agent_mean","agent_std",
                              "random_mean","random_std",
                              "greedy_deadline_mean","greedy_deadline_std",
                              "greedy_ratio_mean","greedy_ratio_std",
                              "whittle_mean","whittle_std"]}
    t0 = time.time()

    for ep in range(1, n_episodes + 1):
        ep_env = DeadlineSchedulingEnv(N, M, seed=SEED + ep)
        rollout(ep_env, None, HORIZON, store=True, agent=agent, gamma=DISCOUNT)

        if ep % UPDATE_FREQ == 0:
            agent.update()

        if ep % EVAL_EVERY == 0:
            ev = DeadlineSchedulingEnv(N, M, seed=SEED)
            am, as_ = evaluate(ev, agent,    EVAL_EPS, HORIZON)
            rm, rs  = evaluate(ev, rand_pol, EVAL_EPS, HORIZON, use_env=True)
            dm, ds  = evaluate(ev, dead_pol, EVAL_EPS, HORIZON, use_env=True)
            gm, gs  = evaluate(ev, ratio_pol,EVAL_EPS, HORIZON, use_env=True)
            wm, ws = evaluate(ev, whittle_pol, EVAL_EPS, HORIZON, use_env=True)

            hist["episodes"].append(ep)
            hist["agent_mean"].append(am);            hist["agent_std"].append(as_)
            hist["random_mean"].append(rm);           hist["random_std"].append(rs)
            hist["greedy_deadline_mean"].append(dm);  hist["greedy_deadline_std"].append(ds)
            hist["greedy_ratio_mean"].append(gm);     hist["greedy_ratio_std"].append(gs)
            hist["whittle_mean"].append(wm)
            hist["whittle_std"].append(ws)

            if verbose:
                print(f"[Ep {ep:5d}/{n_episodes}] "
                    f"Agent:{am:7.1f}±{as_:.1f}  "
                    f"Rand:{rm:6.1f}  "
                    f"Deadline:{dm:6.1f}  "
                    f"Ratio:{gm:6.1f}  "
                    f"Whittle:{wm:6.1f}  "
                    f"τ={agent.tau:.1f}  "
                    f"({time.time()-t0:.0f}s)")

    return hist, agent


def plot_results(hist, N, M, save_path=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return None

    eps = hist["episodes"]
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    clr = {"agent":"#58a6ff","random":"#f0883e","deadline":"#3fb950","ratio":"#bc8cff",
    "whittle":"#ff4d6d"}

    def band(label, ms, ss, c, ls="-"):
        m, s = np.array(ms), np.array(ss)
        ax.plot(eps, m, label=label, color=c, lw=2, ls=ls)
        ax.fill_between(eps, m-s, m+s, color=c, alpha=0.12)

    band("Ranking-PG (ours)",   hist["agent_mean"],           hist["agent_std"],           clr["agent"])
    band("Random",               hist["random_mean"],           hist["random_std"],           clr["random"],   "--")
    band("Greedy-Deadline",      hist["greedy_deadline_mean"],  hist["greedy_deadline_std"],  clr["deadline"], "--")
    band("Greedy-Ratio (B/D+1)", hist["greedy_ratio_mean"],     hist["greedy_ratio_std"],     clr["ratio"],    "--")
    band("Whittle Index",
     hist["whittle_mean"],
     hist["whittle_std"],
     clr["whittle"],
     "-.")

    ax.set_xlabel("Training Episodes", color="#8b949e", fontsize=12)
    ax.set_ylabel("Discounted Return", color="#8b949e", fontsize=12)
    ax.set_title(f"Deadline Scheduling  N={N}, M={M}  |  Ranking-PG vs Baselines",
                 color="#e6edf3", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#8b949e")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    ax.legend(framealpha=0.2, labelcolor="#e6edf3",
              facecolor="#161b22", edgecolor="#30363d", fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Plot saved → {save_path}")
    return fig


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--N",        type=int,  default=10)
    p.add_argument("--M",        type=int,  default=1)
    p.add_argument("--episodes", type=int,  default=2000)
    p.add_argument("--no-plot",  action="store_true")
    args = p.parse_args()

    print(f"\n{'='*60}")
    print(f"  Ranking-PG  |  N={args.N}, M={args.M}  |  {args.episodes} eps")
    print(f"{'='*60}\n")

    hist, agent = train(N=args.N, M=args.M, n_episodes=args.episodes)

    if not args.no_plot:
        plot_results(hist, args.N, args.M,
                     save_path=f"results_N{args.N}_M{args.M}.png")
