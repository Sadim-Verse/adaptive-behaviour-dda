# scripts/plots/plot_figure5_delta_sensitivity.py
"""
Figure 5 — Delta Sensitivity Curves with Seed-Level Variance
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('./logs/figures', exist_ok=True)

# ── Load data ──
steady_csv = "./logs/delta_sensitivity/delta_sensitivity_raw_seeds.csv"
if not os.path.exists(steady_csv):
    print("Data file not found. Skipping.")
    exit()

df = pd.read_csv(steady_csv)

# ── Create figure ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

seed_colors = {
    42: '#1f77b4',      # blue
    105: '#ff7f0e',     # orange
    777: '#2ca02c',     # green
    999: '#d62728',     # red
    2026: '#9467bd'     # purple
}

for i, env in enumerate(['single', 'multi']):
    ax = axes[i]
    env_df = df[df['environment'] == env]
    
    # ── Plot each seed as a thin line with markers ──
    for seed in env_df['seed'].unique():
        seed_df = env_df[env_df['seed'] == seed]
        ax.plot(
            seed_df['delta'],
            seed_df['fzc'],
            marker='o',
            linestyle='-',
            linewidth=1.2,
            markersize=4,
            color=seed_colors[seed],
            alpha=0.7,
            label=f'Seed {seed}'
        )
    
    # ── Plot mean line (thick, black) ──
    mean_df = env_df.groupby('delta')['fzc'].mean().reset_index()
    ax.plot(
        mean_df['delta'],
        mean_df['fzc'],
        marker='s',
        linestyle='-',
        linewidth=3,
        color='black',
        label='Mean (5 seeds)',
        alpha=0.9
    )
    
    ax.set_xlabel('Delta (Δd)', fontsize=12)
    ax.set_ylabel('FZC (%)', fontsize=12)
    ax.set_title(f'{env.upper()} Environment', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(40, 100)

fig.suptitle('Figure 5 — Delta Sensitivity Curves (Steady Mode)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('./logs/figures/figure5_delta_sensitivity.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure 5 saved to ./logs/figures/figure5_delta_sensitivity.png")