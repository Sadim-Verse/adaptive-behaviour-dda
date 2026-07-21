# scripts/plots/plot_no_dda_comparison.py
"""
Figure 7 — Static vs Adaptive Difficulty Comparison
Compares Full System (DDA) vs No DDA for DQN, DDQN, and PPO in the Single environment.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ── CREATE OUTPUT DIRECTORY ──
os.makedirs('./logs/figures', exist_ok=True)

# ── DATA FROM ABLATION RESULTS ──
algorithms = ['DQN', 'DDQN', 'PPO']

# Full System (DDA) — from Table 3
full_system = {
    'DQN':  {'mean_reward': 613.5,  'fzc': 56.3},
    'DDQN': {'mean_reward': 788.6, 'fzc': 68.3},
    'PPO':  {'mean_reward': 678.8, 'fzc': 78.9}
}

# No DDA (Fixed Speed 6.0) — from ablation results
no_dda = {
    'DQN':  {'mean_reward': 473.7,  'fzc': 20.8},
    'DDQN': {'mean_reward': 955.1,  'fzc': 83.3},
    'PPO':  {'mean_reward': 358.2,  'fzc': 0.0}
}

# ── CREATE FIGURE ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

x = np.arange(len(algorithms))
width = 0.35

# --- Mean Reward ---
axes[0].bar(
    x - width/2,
    [full_system[a]['mean_reward'] for a in algorithms],
    width,
    label='Full System (DDA)',
    color='#2ecc71',
    alpha=0.8,
    edgecolor='black',
    linewidth=0.5
)
axes[0].bar(
    x + width/2,
    [no_dda[a]['mean_reward'] for a in algorithms],
    width,
    label='No DDA (Fixed Speed 6.0)',
    color='#e74c3c',
    alpha=0.8,
    edgecolor='black',
    linewidth=0.5
)

axes[0].set_xlabel('Algorithm', fontsize=12)
axes[0].set_ylabel('Mean Episode Reward', fontsize=12)
axes[0].set_title('Mean Reward Comparison', fontsize=12, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(algorithms, fontsize=11)
axes[0].legend(loc='upper right', fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0, 1400)

# Add value labels on bars
for i, algo in enumerate(algorithms):
    axes[0].text(
        i - width/2,
        full_system[algo]['mean_reward'] + 30,
        f"{full_system[algo]['mean_reward']:.0f}",
        ha='center',
        va='bottom',
        fontsize=9,
        fontweight='bold'
    )
    axes[0].text(
        i + width/2,
        no_dda[algo]['mean_reward'] + 30,
        f"{no_dda[algo]['mean_reward']:.0f}",
        ha='center',
        va='bottom',
        fontsize=9,
        fontweight='bold'
    )

# --- FZC ---
axes[1].bar(
    x - width/2,
    [full_system[a]['fzc'] for a in algorithms],
    width,
    label='Full System (DDA)',
    color='#2ecc71',
    alpha=0.8,
    edgecolor='black',
    linewidth=0.5
)
axes[1].bar(
    x + width/2,
    [no_dda[a]['fzc'] for a in algorithms],
    width,
    label='No DDA (Fixed Speed 6.0)',
    color='#e74c3c',
    alpha=0.8,
    edgecolor='black',
    linewidth=0.5
)

axes[1].set_xlabel('Algorithm', fontsize=12)
axes[1].set_ylabel('Flow Zone Consistency (FZC %)', fontsize=12)
axes[1].set_title('FZC Comparison', fontsize=12, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(algorithms, fontsize=11)
axes[1].legend(loc='upper right', fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, 100)

# Add value labels on bars
for i, algo in enumerate(algorithms):
    axes[1].text(
        i - width/2,
        full_system[algo]['fzc'] + 3,
        f"{full_system[algo]['fzc']:.1f}%",
        ha='center',
        va='bottom',
        fontsize=9,
        fontweight='bold'
    )
    axes[1].text(
        i + width/2,
        no_dda[algo]['fzc'] + 3,
        f"{no_dda[algo]['fzc']:.1f}%",
        ha='center',
        va='bottom',
        fontsize=9,
        fontweight='bold'
    )

fig.suptitle(
    'Figure 7 — Static vs Adaptive Difficulty Comparison (Single Environment)',
    fontsize=14,
    fontweight='bold'
)
plt.tight_layout()
plt.savefig('./report_figures/figure7_static_vs_adaptive.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Figure 7 saved to ./report_figures/figure7_static_vs_adaptive.png")