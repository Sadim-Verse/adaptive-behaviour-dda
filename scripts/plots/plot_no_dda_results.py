# scripts/plot_no_dda_results.py
"""
Extract and plot No DDA ablation results for DQN, DDQN, and PPO.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Configuration ──
BASE_DIR = "./logs/ablations/no_dda"
ALGOS = ["dqn", "ddqn", "ppo"]
SEED = 42

# Flow zone bounds for Single environment
FLOW_LOW, FLOW_HIGH = 600, 1200

def compute_metrics(folder_path):
    """Extract mean reward and FZC from a run folder."""
    rewards_file = os.path.join(folder_path, "episode_rewards.npy")
    rolling_file = os.path.join(folder_path, "rolling_avg.npy")
    
    if not os.path.exists(rewards_file) or not os.path.exists(rolling_file):
        return None
    
    rewards = np.load(rewards_file)
    rolling_avgs = np.load(rolling_file)
    
    mean_reward = np.mean(rewards)
    reward_std = np.std(rewards)
    
    # Compute FZC
    valid_avgs = rolling_avgs[9:]  # After warm-up (10 episodes)
    in_zone = np.sum((valid_avgs >= FLOW_LOW) & (valid_avgs <= FLOW_HIGH))
    fzc = (in_zone / len(valid_avgs)) * 100.0
    
    return {"mean_reward": mean_reward, "reward_std": reward_std, "fzc": fzc}

print("=" * 60)
print("NO DDA ABLATION RESULTS (SINGLE ENVIRONMENT)")
print("=" * 60)

results = {}
for algo in ALGOS:
    folder = f"single_{algo}_delta0p05_seed{SEED}_abl_no_dda"
    path = os.path.join(BASE_DIR, folder)
    
    if os.path.exists(path):
        metrics = compute_metrics(path)
        if metrics:
            results[algo] = metrics
            print(f"{algo.upper():<8} | Mean Reward: {metrics['mean_reward']:.1f} ± {metrics['reward_std']:.1f} | FZC: {metrics['fzc']:.1f}%")
        else:
            print(f"{algo.upper():<8} | ⚠️ Missing data files")
    else:
        print(f"{algo.upper():<8} | ⚠️ Folder not found: {path}")

# ── Plot comparison: No DDA vs Full System ──

# Data from Table 3 (Full System)
full_system = {
    "dqn": {"mean_reward": 975.5, "fzc": 52.9},
    "ddqn": {"mean_reward": 1174.3, "fzc": 63.8},
    "ppo": {"mean_reward": 1005.1, "fzc": 72.9}
}

# No DDA data from ablation (use results above)
no_dda = {}
for algo, data in results.items():
    no_dda[algo] = {"mean_reward": data["mean_reward"], "fzc": data["fzc"]}

# Create comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
x = np.arange(len(ALGOS))
width = 0.35

# Mean Reward
axes[0].bar(x - width/2, [full_system[a]["mean_reward"] for a in ALGOS], width, label='Full System (DDA)', color='#2ecc71', alpha=0.7)
axes[0].bar(x + width/2, [no_dda.get(a, {}).get("mean_reward", 0) for a in ALGOS], width, label='No DDA (Fixed)', color='#e74c3c', alpha=0.7)
axes[0].set_xlabel('Algorithm')
axes[0].set_ylabel('Mean Reward')
axes[0].set_title('Mean Reward Comparison')
axes[0].set_xticks(x)
axes[0].set_xticklabels([a.upper() for a in ALGOS])
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# FZC
axes[1].bar(x - width/2, [full_system[a]["fzc"] for a in ALGOS], width, label='Full System (DDA)', color='#2ecc71', alpha=0.7)
axes[1].bar(x + width/2, [no_dda.get(a, {}).get("fzc", 0) for a in ALGOS], width, label='No DDA (Fixed)', color='#e74c3c', alpha=0.7)
axes[1].set_xlabel('Algorithm')
axes[1].set_ylabel('FZC (%)')
axes[1].set_title('FZC Comparison')
axes[1].set_xticks(x)
axes[1].set_xticklabels([a.upper() for a in ALGOS])
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.suptitle('Figure 7 — Static vs Adaptive Difficulty Comparison (Single Environment)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('./logs/figures/figure7_static_vs_adaptive.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Figure 7 saved to ./logs/figures/figure7_static_vs_adaptive.png")