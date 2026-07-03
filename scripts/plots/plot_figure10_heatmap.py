# scripts/plots/plot_figure10_ratio_heatmap.py
"""
Figure 10 — Generalisation Performance Across Environments
Heatmap showing Multi/Single ratios for DDQN and PPO.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

os.makedirs('./logs/figures', exist_ok=True)

# ── Data from Table 3 ──
data = {
    'Algorithm': ['DDQN', 'PPO'],
    'Mean Reward Ratio': [336.6/1174.3, 352.7/1005.1],
    'FZC Ratio': [85.6/63.8, 72.3/72.9]
}

df = pd.DataFrame(data)
df = df.set_index('Algorithm')
df = df[['Mean Reward Ratio', 'FZC Ratio']]  # Reorder columns

print("=" * 60)
print("GENERALISATION RATIOS (Multi / Single)")
print("=" * 60)
print(df.round(3))
print("=" * 60)

# ── Create figure ──
fig, ax = plt.subplots(figsize=(8, 5))

# Custom colormap: red (low) → yellow → green (high)
cmap = sns.diverging_palette(10, 130, s=80, l=55, as_cmap=True)

# Create heatmap with annotations
sns.heatmap(
    df,
    annot=True,
    fmt='.3f',
    cmap=cmap,
    center=1.0,
    vmin=0.0,
    vmax=1.5,
    cbar_kws={'label': 'Generalisation Ratio (Multi / Single)'},
    linewidths=0.5,
    ax=ax,
    square=True,
    annot_kws={'size': 14, 'weight': 'bold'}
)

ax.set_title('Figure 10 — Generalisation Performance Across Environments', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Metric', fontsize=12)
ax.set_ylabel('Algorithm', fontsize=12)

# ── Add interpretation note ──
note = (
    "DDQN: FZC generalises well (1.34) but mean reward drops (0.29)\n"
    "PPO: Both metrics drop moderately (0.35, 0.99)"
)
ax.text(0.5, -0.18, note,
        transform=ax.transAxes, fontsize=9, ha='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ── Add value labels with interpretation ──
# Highlight DDQN's FZC > 1.0
for i, algo in enumerate(df.index):
    for j, metric in enumerate(df.columns):
        val = df.loc[algo, metric]
        color = 'white' if val > 1.0 else 'black'
        ax.text(j + 0.5, i + 0.5, f'{val:.3f}',
                ha='center', va='center', fontsize=14, fontweight='bold',
                color=color)

plt.tight_layout()
plt.savefig('./logs/figures/figure10_ratio_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Figure 10 saved to ./logs/figures/figure10_ratio_heatmap.png")