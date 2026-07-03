# scripts/analyze_ablations.py
"""
Analyse ablation results and produce Table 5.
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = "./logs/ablations"
ABLATIONS = ["no_dda", "fixed_difficulty", "no_rolling_window", "no_reward_smoothing"]
ENVS = ["single", "multi"]
SEEDS = [42, 105, 777, 999, 2026]

# Flow zone bounds
FLOW_ZONES = {"single": (600.0, 1200.0), "multi": (220.0, 470.0)}

def get_folder_name(ablation, env, seed):
    """
    Get the actual folder name for a given ablation, environment, and seed.
    
    Folder naming convention:
    - no_dda:         {env}_ddqn_delta0p05_seed{seed}_abl_no_dda
    - fixed_difficulty: {env}_ddqn_delta0p05_seed{seed}_abl_fixed_difficulty
    - no_rolling_window: {env}_ddqn_delta0p05_seed{seed}_abl_no_rolling_window
    - no_reward_smoothing: {env}_ddqn_delta0p05_seed{seed}_abl_no_smoothing
    """
    if ablation == "no_reward_smoothing":
        return f"{env}_ddqn_delta0p05_seed{seed}_abl_no_smoothing"
    else:
        return f"{env}_ddqn_delta0p05_seed{seed}_abl_{ablation}"

def compute_fzc(rolling_avgs, flow_low, flow_high, window_size=10):
    if len(rolling_avgs) < window_size:
        return 0.0
    valid_avgs = rolling_avgs[window_size - 1:]
    in_zone = np.sum((valid_avgs >= flow_low) & (valid_avgs <= flow_high))
    return (in_zone / len(valid_avgs)) * 100.0

def compute_metrics(folder_path, env):
    rewards_file = os.path.join(folder_path, "episode_rewards.npy")
    rolling_file = os.path.join(folder_path, "rolling_avg.npy")
    
    if not os.path.exists(rewards_file) or not os.path.exists(rolling_file):
        return None
    
    rewards = np.load(rewards_file)
    rolling_avgs = np.load(rolling_file)
    
    flow_low, flow_high = FLOW_ZONES[env]
    
    mean_reward = np.mean(rewards)
    reward_std = np.std(rewards)
    fzc = compute_fzc(rolling_avgs, flow_low, flow_high)
    
    return {"mean_reward": mean_reward, "reward_std": reward_std, "fzc": fzc}

print("=" * 80)
print("TABLE 5 — ABLATION STUDY RESULTS")
print("=" * 80)

# First, check what folders actually exist
print("\n🔍 Checking folder structure...")
for ablation in ABLATIONS:
    ablation_path = os.path.join(BASE_DIR, ablation)
    if os.path.exists(ablation_path):
        print(f"  ✅ {ablation_path} exists")
    else:
        print(f"  ❌ {ablation_path} does not exist")

results = []

for env in ENVS:
    print(f"\n{'─'*80}")
    print(f"ENVIRONMENT: {env.upper()}")
    print(f"{'─'*80}")
    
    for ablation in ABLATIONS:
        rewards_list = []
        fzcs_list = []
        ablation_label = ablation.replace("_", " ").title()
        
        for seed in SEEDS:
            folder = get_folder_name(ablation, env, seed)
            path = os.path.join(BASE_DIR, ablation, folder)
            
            # Check if the folder exists first
            if not os.path.exists(path):
                print(f"  ⚠️ Folder not found: {path}")
                continue
            
            metrics = compute_metrics(path, env)
            if metrics:
                rewards_list.append(metrics["mean_reward"])
                fzcs_list.append(metrics["fzc"])
            else:
                print(f"  ⚠️ Missing files in: {path}")
        
        if rewards_list:
            mean_rew = np.mean(rewards_list)
            std_rew = np.std(rewards_list)
            mean_fzc = np.mean(fzcs_list)
            std_fzc = np.std(fzcs_list)
            
            results.append({
                "ablation": ablation,
                "environment": env,
                "mean_reward": mean_rew,
                "std_reward": std_rew,
                "fzc": mean_fzc,
                "std_fzc": std_fzc
            })
            
            print(f"{ablation_label:<25} | "
                  f"{mean_rew:>8.1f} ± {std_rew:<5.1f} | "
                  f"{mean_fzc:>6.1f}% ± {std_fzc:<4.1f}%")
        else:
            print(f"{ablation_label:<25} | ⚠️ No data found")

print("\n" + "=" * 80)
print("TABLE 5 — FORMATTED FOR THESIS")
print("=" * 80)

# Print formatted table
for env in ENVS:
    print(f"\n{env.upper()} ENVIRONMENT:")
    print("-" * 70)
    print(f"{'Configuration':<25} | {'Mean Reward':<18} | {'FZC (%)':<15}")
    print("-" * 70)
    
    # Add Full System row (from Table 3)
    if env == "single":
        full_system_mean = "1174.3"
        full_system_std = "42.9"
        full_system_fzc = "63.8%"
        full_system_fzc_std = "4.8%"
    else:  # multi
        full_system_mean = "337.7"
        full_system_std = "8.0"
        full_system_fzc = "85.6%"
        full_system_fzc_std = "2.3%"
    
    print(f"{'Full System':<25} | {full_system_mean} ± {full_system_std:<5} | {full_system_fzc} ± {full_system_fzc_std}")
    
    for row in results:
        if row["environment"] == env:
            label = row["ablation"].replace("_", " ").title()
            print(f"{label:<25} | "
                  f"{row['mean_reward']:>8.1f} ± {row['std_reward']:<5.1f} | "
                  f"{row['fzc']:>6.1f}% ± {row['std_fzc']:<4.1f}%")
    print("-" * 70)

print("=" * 80)