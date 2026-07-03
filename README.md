# Adaptive Behaviour DDA System

**Reinforcement Learning-based Adaptive Behaviour System for Dynamic Game Difficulty Adjustment**

*Final Year Project — B.Sc. Computer Science*

---

## Overview

This project implements a reinforcement learning-based system that evaluates agent robustness under dynamic difficulty adjustment. The system consists of:

- **Two game environments**: Single-enemy (`ChaseEnv`) and Multi-enemy (`MultiChaseEnv`)
- **Three RL algorithms**: DQN, DDQN (van Hasselt implementation), and PPO
- **Rule-based DDA Controller**: Adjusts enemy speed in real time to maintain agent performance within a Flow Zone
- **Comprehensive evaluation**: Flow Zone Consistency, Adaptation Latency, Delta Sensitivity, and Ablation Studies

---

## Key Features

- ✅ Custom Pygame environments with Gymnasium interface
- ✅ Corrected DDQN implementation (van Hasselt et al., 2016)
- ✅ Rule-based DDA Controller with proportional feedback
- ✅ Spike Protocol for measuring Adaptation Latency
- ✅ Multi-seed validation (5 random seeds)
- ✅ Complete ablation studies
- ✅ CPU-only training (no GPU required)

---

## Results Summary

| Environment | Algorithm | Mean Reward | FZC (%) | Adaptation Latency (eps) |
| :--- | :--- | :--- | :--- | :--- |
| **Single** | DDQN | 1174.3 ± 42.9 | 63.8% ± 4.8% | 11.6 ± 2.1 |
| **Single** | PPO | 1005.1 ± 18.0 | 72.9% ± 1.1% | 120.8 ± 31.4 |
| **Multi** | DDQN | 336.6 ± 7.4 | 85.6% ± 2.3% | 207.6 ± 148.2 |
| **Multi** | PPO | 352.7 ± 5.5 | 72.3% ± 5.6% | 131.6 ± 14.2 |

**Key Findings:**
- DDQN is the most robust algorithm — fastest recovery and highest FZC.
- The DDA Controller is essential for PPO in stochastic environments.
- The rolling window is critical for FZC (drops from 63.8% to 18.8% without it).
- The responsiveness-stability trade-off is confirmed in spike mode.

---

## Project Structure

```
Adaptive_Agent/
├── envs/                          # Game environments
│   ├── chase_env.py               # Single-enemy environment
│   └── multi_chase_env.py         # Three-enemy environment
├── training/
│   ├── train_multi_env.py         # Unified training/evaluation script
│   ├── dda_controller.py          # DDA Controller logic
│   └── ddqn.py                    # Corrected DDQN implementation
├── scripts/                       # Analysis and plotting scripts
│   ├── compile_results.py         # Main benchmark results
│   ├── statistical_tests.py       # Statistical significance
│   ├── analyze_ablations.py       # Ablation study results
│   ├── analyze_spike_delta.py     # Spike-mode delta sensitivity
│   └── calibrate_flow_zones.py    # Flow Zone calibration
├── visualising/
│   └── visualize_agent.py         # Visualise agent behaviour
├── logs/                          # Logs and results (not included in repo)
└── README.md
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/adaptive-behaviour-dda.git
cd adaptive-behaviour-dda
```

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running Experiments

### Train an Agent

```bash
# Train DDQN on Single environment (1M steps)
python -m training.train_multi_env --algo ddqn --env single --timesteps 1000000

# Train DDQN on Multi environment (6M steps)
python -m training.train_multi_env --algo ddqn --env multi --timesteps 6000000
```

### Evaluate an Agent (Steady Mode)

```bash
python -m training.train_multi_env --algo ddqn --env single --resume ./logs/experiments/ddqn/single_ddqn_delta0p05_seed42/best_model.zip --delta 0.05 --seed 42 --max_speed 9.0
```

### Evaluate with Spike Protocol

```bash
python -m training.train_multi_env --algo ddqn --env single --resume ./logs/experiments/ddqn/single_ddqn_delta0p05_seed42/best_model.zip --delta 0.05 --seed 42 --max_speed 9.0 --spike 250
```

### Run All Evaluations (30 Runs)

```bash
python -m scripts.run_all_evaluations --mode steady
python -m scripts.run_all_evaluations --mode spike
```

### Generate Results

```bash
python -m scripts.compile_results           # Table 3
python -m scripts.statistical_tests         # Table 4
python -m scripts.analyze_ablations         # Table 5
python -m scripts.analyze_spike_delta       # Table 6 (Spike mode)
python -m scripts.analyze_delta_sensitivity # Table 6 (Steady mode)
```

### Visualisation

```bash
# Visualise a trained agent
python -m visualising.visualize_agent --algo ddqn --env single --model ./logs/experiments/ddqn/single_ddqn_delta0p05_seed42/best_model.zip --speed 2.0
```