# scripts/analyze_delta_sensitivity.py
"""
Read and display steady-mode delta sensitivity results.
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = "./logs/delta_sensitivity"

# Load the data
agg_df = pd.read_csv(os.path.join(BASE_DIR, "delta_sensitivity_aggregated.csv"))

print("=" * 90)
print("STEADY-MODE DELTA SENSITIVITY RESULTS")
print("=" * 90)
print(f"  Environments: {agg_df['environment'].unique().tolist()}")
print("=" * 90)

for env in ["single", "multi"]:
    print(f"\n{'─'*90}")
    print(f"ENVIRONMENT: {env.upper()}")
    print(f"{'─'*90}")
    
    env_df = agg_df[agg_df["environment"] == env]
    
    print(f"{'Delta':<8} | {'Mean Reward':<18} | {'FZC (%)':<14} | {'Oscillation':<14} | {'Latency (eps)':<12}")
    print("-" * 90)
    
    for _, row in env_df.iterrows():
        lat_str = "N/A"
        print(f"{row['delta']:<8.2f} | "
              f"{row['mean_reward']:>8.1f} ± {row['reward_std_across_seeds']:<5.1f} | "
              f"{row['fzc_mean']:>8.1f}% ± {row['fzc_std']:<4.1f}% | "
              f"{row['oscillation_freq_mean']:>8.4f} ± {row['oscillation_freq_std']:<.4f} | "
              f"{lat_str}")
    print("-" * 90)

print("\n" + "=" * 90)
print("KEY OBSERVATIONS")
print("=" * 90)