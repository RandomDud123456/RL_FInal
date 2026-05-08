# Restless Multi-Arm Bandits: Bypassing Scalar Rewards with Direct Index Policies

This repository contains advanced Reinforcement Learning implementations for solving **Restless Multi-Armed Bandit (RMAB)** problems. The project focuses on state-ranking policies, index learning, and scalable scheduling algorithms for complex environments.

## Key Algorithms

### 1. H-SRAC (Homogeneous State-Ranking Actor-Critic)
A "Universal Ranker" approach that bridges Deep RL with Whittle Index theory.
- **Global Actor ($\pi_\theta$):** Maps states to priority scores (Logits), ensuring homogeneous treatment across arms.
- **Newton-Balanced Sigmoid:** A differentiable budget regulator that shifts scores using a global bias to enforce the budget constraint $M$.
- **Stochastic Sensitivity ($\lambda$):** Uses variable sharpness to ensure robust ranking convergence.

### 2. Enhanced H-SRAC Variants
- **BT-SRAC:** Integrates **Bradley-Terry** pairwise learning to shift from absolute logit regression to relative preference learning, stabilizing state-ranking.
- **BT-PPO-SRAC:** Adds **PPO Clipping** to the ranking ratio, preventing violent logit fluctuations and ensuring a stable trust-region for policy updates.

### 3. NEURWIN
A ranking-based policy designed for fast convergence and scalability.
- Uses the **Plackett-Luce** ranking model.
- Implements the **Gumbel-Max trick** for differentiable top-$M$ sampling.
- Decouples state-hierarchy learning from budget constraints.

### 4. Index-Based Policies (MC-QGI & MC-QWI)
- **Q-Gittins Index (QGI):** Monte Carlo based learning for rested bandits.
- **Q-Whittle Index (QWI):** Neural network based approximation of the Whittle Index for restless bandits.

---

## Problem Settings

### Deadline Scheduling
Minimizes total penalty from missed deadlines.
- **State (B, D):** $B$ is remaining workload, $D$ is time until deadline.
- **Goal:** Prioritize arms to prevent $D$ reaching 1 while $B > 0$.

### Machine Repair
Optimizes maintenance schedules for a fleet of machines.
- **State:** Wear/tear level of each machine.
- **Goal:** Minimize downtime and repair costs.

### Wireless Scheduling
Allocates communication channels based on stochastic quality.
- **State:** Channel quality index.
- **Goal:** Maximize throughput under limited bandwidth.

---

## Project Structure

```text
RL_FInal/
├── Final_Presentation/
│   ├── Policy_Gradient_Algos/
│   │   ├── HSRAC/                 # Core H-SRAC, BT-SRAC, BT-PPO-SRAC
│   │   │   ├── hsrac.py
│   │   │   ├── hsrac_bt.py
│   │   │   └── hsrac_bt_ppo.py
│   │   └── Actor-Critic/
│   │       └── Code/              # Deadline Env & PPO Baselines
│   │           ├── deadline_env.py
│   │           ├── train.py
│   │           └── agent.py
│   ├── MC-QGI-QWI/                # Q-Index implementations
│   │   ├── qwi.py
│   │   ├── qgi.py
│   │   └── environment.py         # Standard RMAB Environment
│   └── DOPL/                      # Deep Online Policy Learning
└── main.tex                       # Project Presentation Source
```

---

## Installation & Usage

### Prerequisites
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib

### Running Training
To train the H-SRAC agent on the deadline scheduling environment:
```bash
cd Final_Presentation/Policy_Gradient_Algos/Actor-Critic/Code
python train.py
```

To run the Q-Whittle Index implementation:
```bash
cd Final_Presentation/MC-QGI-QWI
python qwi.py
```

---

## Performance Summary
- **NEURWIN** demonstrates the fastest and most stable convergence across both small-scale ($N=4$) and large-scale ($N=100$) environments.
- **H-SRAC** shows strong performance in small settings but requires the Bradley-Terry and PPO enhancements to scale effectively to $N=100$.
- **BT-PPO-SRAC** provides superior stability compared to vanilla H-SRAC by constraining relative preference shifts.

---

## License
This project is part of the BetaStay RL research initiative.
