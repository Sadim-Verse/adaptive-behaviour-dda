# scripts/plot_delta_sensitivity.py
"""
Plot delta sensitivity results from steady-state runs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv("./logs/delta_sensitivity/delta_sensitivity_raw_seeds.csv")

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Delta Sensitivity Analysis: FZC vs Delta (Steady Mode)", fontsize=14, fontweight='bold')

# Colour map for seeds
seed_colours = {42: '#1f77b4', 105: '#ff7f0e', 777: '#2ca02c', 999: '#d62728', 2026: '#9467bd'}

for i, env in enumerate(["single", "multi"]):
    ax = axes[i]
    env_df = df[df["environment"] == env]
    
    # Plot each seed
    for seed in env_df["seed"].unique():
        seed_df = env_df[env_df["seed"] == seed]
        ax.plot(seed_df["delta"], seed_df["fzc"], 
                marker='o', linewidth=2, markersize=8,
                label=f"Seed {seed}", color=seed_colours[seed])
    
    # Add mean line
    mean_df = env_df.groupby("delta")["fzc"].mean().reset_index()
    ax.plot(mean_df["delta"], mean_df["fzc"], 
            marker='s', linewidth=3, markersize=10,
            label="Mean (5 seeds)", color='black', linestyle='--')
    
    ax.set_xlabel("Delta (Δd)", fontsize=12)
    ax.set_ylabel("FZC (%)", fontsize=12)
    ax.set_title(f"{env.upper()} Environment", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(50, 100)

plt.tight_layout()
plt.savefig("./logs/delta_sensitivity/delta_sensitivity_steady_plots.png", dpi=300, bbox_inches="tight")
print("✅ Plot saved to: ./logs/delta_sensitivity/delta_sensitivity_steady_plots.png")
plt.show()