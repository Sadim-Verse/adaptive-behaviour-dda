# scripts/plots/plot_figure9_correlation_fixed.py
"""
Figure 9 — Correlation: Difficulty vs Reward (Fixed)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('./logs/figures', exist_ok=True)

# ── Load data ──
rewards_path = "./logs/delta_sensitivity_spike/single_ddqn_delta0p05_seed42_spike/episode_rewards.npy"
speed_path = "./logs/delta_sensitivity_spike/single_ddqn_delta0p05_seed42_spike/speed_history.npy"

rewards = np.load(rewards_path)
speed = np.load(speed_path)

# ── Add jitter to speed values ──
np.random.seed(42)
jitter = np.random.normal(0, 0.05, size=len(speed))
speed_jittered = speed + jitter

fig, ax = plt.subplots(figsize=(10, 6))

# Scatter with alpha for transparency
ax.scatter(speed_jittered, rewards, alpha=0.15, s=8, color='#2c3e50')

# ── Overlay binned mean line ──
bins = np.arange(2.0, 9.5, 0.5)
bin_centers = []
bin_means = []
bin_stds = []

for i in range(len(bins)-1):
    mask = (speed >= bins[i]) & (speed < bins[i+1])
    if np.sum(mask) > 0:
        bin_centers.append((bins[i] + bins[i+1]) / 2)
        bin_means.append(np.mean(rewards[mask]))
        bin_stds.append(np.std(rewards[mask]))

ax.plot(bin_centers, bin_means, 'r-', linewidth=2.5, label='Binned Mean')
ax.fill_between(bin_centers, 
                np.array(bin_means) - np.array(bin_stds),
                np.array(bin_means) + np.array(bin_stds),
                alpha=0.2, color='red', label='±1 Std')

ax.axhline(600, color='green', linestyle='--', alpha=0.7, label='Flow Low (600)')
ax.axhline(1200, color='blue', linestyle='--', alpha=0.7, label='Flow High (1200)')

ax.set_xlabel('Enemy Speed', fontsize=12)
ax.set_ylabel('Episode Reward', fontsize=12)
ax.set_title('Figure 9 — Correlation: Difficulty vs Reward', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(1.8, 9.2)
ax.set_ylim(-100, 2100)

plt.tight_layout()
plt.savefig('./logs/figures/figure9_correlation_fixed.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure 9 saved to ./logs/figures/figure9_correlation_fixed.png")