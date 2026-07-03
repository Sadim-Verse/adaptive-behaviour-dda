# scripts/verify_table6.py
"""
Crosscheck Table 6 — Delta Sensitivity Analysis values.
Reads data from both steady and spike mode runs and computes all metrics.
"""

import os
import numpy as np
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────── #

ENVS = ["single", "multi"]
SEEDS = [42, 105, 777, 999, 2026]
DELTAS = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]

# Steady mode data location
STEADY_BASE_DIR = "./logs/delta_sensitivity"
SPIKE_BASE_DIR = "./logs/delta_sensitivity_spike"

# Algorithm folder mapping
ALGO_FOLDERS = {
    "dqn": "dqn",
    "ddqn": "ddqn",
    "ppo": "ppo"
}

# Flow zone bounds
FLOW_ZONES = {"single": (600.0, 1200.0), "multi": (220.0, 470.0)}

def compute_fzc(rolling_avgs, flow_low, flow_high, window_size=10):
    if len(rolling_avgs) < window_size:
        return 0.0
    valid_avgs = rolling_avgs[window_size - 1:]
    in_zone = np.sum((valid_avgs >= flow_low) & (valid_avgs <= flow_high))
    return (in_zone / len(valid_avgs)) * 100.0

def compute_oscillation_frequency(speed_history):
    if len(speed_history) < 3:
        return 0.0
    diffs = np.diff(speed_history)
    direction_changes = 0
    for i in range(1, len(diffs)):
        if diffs[i] * diffs[i-1] < 0:
            direction_changes += 1
    return direction_changes / len(speed_history)

def load_adaptation_latency(folder_path):
    """Load adaptation latency from .npy file."""
    latency_file = os.path.join(folder_path, "adaptation_latency.npy")
    if os.path.exists(latency_file):
        latency = np.load(latency_file)
        if len(latency) > 0:
            return float(latency[0])
    return None

def get_metrics_from_folder(folder_path, env):
    """Extract metrics from a single run folder."""
    rewards_file = os.path.join(folder_path, "episode_rewards.npy")
    rolling_file = os.path.join(folder_path, "rolling_avg.npy")
    speed_file = os.path.join(folder_path, "speed_history.npy")
    
    if not (os.path.exists(rewards_file) and os.path.exists(rolling_file) and os.path.exists(speed_file)):
        return None
    
    rewards = np.load(rewards_file)
    rolling_avgs = np.load(rolling_file)
    speed_history = np.load(speed_file)
    
    flow_low, flow_high = FLOW_ZONES[env]
    
    mean_reward = np.mean(rewards)
    reward_std = np.std(rewards)
    fzc = compute_fzc(rolling_avgs, flow_low, flow_high)
    oscillation = compute_oscillation_frequency(speed_history)
    latency = load_adaptation_latency(folder_path)
    
    return {
        "mean_reward": mean_reward,
        "reward_std": reward_std,
        "fzc": fzc,
        "oscillation": oscillation,
        "latency": latency if latency is not None else float('nan')
    }

def load_delta_data(env, delta, mode="steady"):
    """Load data for a specific environment and delta across all seeds."""
    delta_str = str(delta).replace('.', 'p')
    
    if mode == "steady":
        base_dir = STEADY_BASE_DIR
        suffix = ""
    else:  # spike
        base_dir = SPIKE_BASE_DIR
        suffix = "_spike"
    
    rewards_list = []
    fzcs_list = []
    oscillations_list = []
    latencies_list = []
    
    for seed in SEEDS:
        # Build folder path
        if mode == "steady":
            folder = f"{env}_ddqn_delta{delta_str}_seed{seed}"
        else:
            folder = f"{env}_ddqn_delta{delta_str}_seed{seed}_spike"
        
        folder_path = os.path.join(base_dir, folder)
        
        if not os.path.exists(folder_path):
            print(f"  ⚠️ Folder not found: {folder_path}")
            continue
        
        metrics = get_metrics_from_folder(folder_path, env)
        if metrics is None:
            print(f"  ⚠️ Missing files in: {folder_path}")
            continue
        
        rewards_list.append(metrics["mean_reward"])
        fzcs_list.append(metrics["fzc"])
        oscillations_list.append(metrics["oscillation"])
        latencies_list.append(metrics["latency"])
    
    if not rewards_list:
        return None
    
    return {
        "mean_reward_mean": np.mean(rewards_list),
        "mean_reward_std": np.std(rewards_list),
        "fzc_mean": np.mean(fzcs_list),
        "fzc_std": np.std(fzcs_list),
        "oscillation_mean": np.mean(oscillations_list),
        "oscillation_std": np.std(oscillations_list),
        "latency_mean": np.nanmean(latencies_list),
        "latency_std": np.nanstd(latencies_list),
        "n_seeds": len(rewards_list)
    }

print("=" * 90)
print("TABLE 6 VERIFICATION — DELTA SENSITIVITY ANALYSIS")
print("=" * 90)

all_data = {}

for mode in ["steady", "spike"]:
    mode_label = mode.upper()
    print(f"\n{'─'*90}")
    print(f"MODE: {mode_label}")
    print(f"{'─'*90}")
    
    for env in ENVS:
        print(f"\n{env.upper()} ENVIRONMENT:")
        print("-" * 90)
        print(f"{'Delta':<8} | {'Mean Reward':<18} | {'FZC (%)':<14} | {'Oscillation':<14} | {'Latency (eps)':<12}")
        print("-" * 90)
        
        for delta in DELTAS:
            data = load_delta_data(env, delta, mode)
            if data is None:
                print(f"{delta:<8.2f} | ⚠️ No data found")
                continue
            
            lat_str = f"{data['latency_mean']:.1f} ± {data['latency_std']:.1f}" if not np.isnan(data['latency_mean']) else "N/A"
            print(f"{delta:<8.2f} | "
                  f"{data['mean_reward_mean']:>8.1f} ± {data['mean_reward_std']:<5.1f} | "
                  f"{data['fzc_mean']:>8.1f}% ± {data['fzc_std']:<4.1f}% | "
                  f"{data['oscillation_mean']:>8.4f} ± {data['oscillation_std']:<.4f} | "
                  f"{lat_str}")
        print("-" * 90)

print("\n" + "=" * 90)
print("VERIFICATION COMPLETE")
print("=" * 90)
print("Compare the values above with Table 6 in your thesis.")
print("If values match, Table 6 is correct.")
print("=" * 90)