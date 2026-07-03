# scripts/plot_figure2_training_curves.py
"""
Figure 2 — Training Curves Across Algorithms
Uses the actual CSV files exported from TensorBoard.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ── CREATE THE FIGURES DIRECTORY ──
os.makedirs('./logs/figures', exist_ok=True)

# ── FILL IN YOUR FILE PATHS HERE ──
dqn_csv = "./logs/experiments/Rewards_Fig_2/single_dqn_delta0p05_seed42_3.csv"       # Path to your DQN CSV
ddqn_csv = "./logs/experiments/Rewards_Fig_2/single_ddqn_delta0p05_seed42_10.csv"    # Path to your DDQN CSV
ppo_csv = "./logs/experiments/Rewards_Fig_2/single_ppo_delta0p05_seed42_1.csv"       # Path to your PPO CSV

# ── Load data ──
dqn_df = pd.read_csv(dqn_csv)
ddqn_df = pd.read_csv(ddqn_csv)
ppo_df = pd.read_csv(ppo_csv)

# ── Create figure ──
fig, ax = plt.subplots(figsize=(12, 6))

# Plot each algorithm's training curve
ax.plot(dqn_df['Step'], dqn_df['Value'], label='DQN', linewidth=2, color='#1f77b4')
ax.plot(ddqn_df['Step'], ddqn_df['Value'], label='DDQN', linewidth=2, color='#ff7f0e')
ax.plot(ppo_df['Step'], ppo_df['Value'], label='PPO', linewidth=2, color='#2ca02c')

ax.set_xlabel('Training Steps', fontsize=12)
ax.set_ylabel('Mean Episode Reward (Training)', fontsize=12)
ax.set_title('Figure 2 — Training Curves Across Algorithms', fontsize=14, fontweight='bold')

# ── Moved legend to centre right ──
ax.legend(loc='center right', fontsize=10)

ax.grid(True, alpha=0.3)

# Optional: Add a note about the gap between training and evaluation
ax.text(0.02, 0.98, 'Note: Training curves include exploration noise.\nFinal evaluation rewards are significantly higher.',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('./logs/figures/figure2_training_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 2 saved to ./logs/figures/figure2_training_curves.png")