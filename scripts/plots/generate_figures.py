#!/usr/bin/env python3
# generate_figures.py
"""
Generate all figures for the thesis.
Fill in the file paths where indicated by "PLACEHOLDER".

Required packages:
    pip install numpy pandas matplotlib seaborn scipy
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Create output directory
os.makedirs('./logs/figures', exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('colorblind')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# =============================================================================
# FIGURE 2 — Training Curves Across Algorithms
# =============================================================================
def figure2_training_curves():
    """
    Needs: TensorBoard event files or exported CSV of episode reward vs steps.
    Placeholder: path to a CSV with columns: steps, dqn, ddqn, ppo
    """
    # --- FILL THIS FILE PATH ---
    data_path = "PLACEHOLDER"  # e.g., './logs/training_curves.csv'
    
    if not os.path.exists(data_path):
        print("Figure 2: Data file not found. Skipping.")
        return
    
    df = pd.read_csv(data_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['steps'], df['dqn'], label='DQN', linewidth=2)
    ax.plot(df['steps'], df['ddqn'], label='DDQN', linewidth=2)
    ax.plot(df['steps'], df['ppo'], label='PPO', linewidth=2)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Mean Episode Reward', fontsize=12)
    ax.set_title('Figure 2 — Training Curves Across Algorithms', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./logs/figures/figure2_training_curves.png', dpi=300)
    plt.close()
    print("Figure 2 saved.")

# =============================================================================
# FIGURE 3 — Reward Stability Under Dynamic Difficulty
# =============================================================================
def figure3_reward_stability():
    """
    Needs: rolling_avg.npy for a spike run, flow bounds, spike episode, latency.
    Placeholder: path to rolling_avg.npy and adaptation_latency.npy.
    """
    # --- FILL THESE FILE PATHS ---
    rolling_avg_path = "./logs/delta_sensitivity_spike/multi_ddqn_delta0p05_seed42_spike/rolling_avg.npy"  # e.g., './logs/delta_sensitivity_spike/single_ddqn_delta0p05_seed42_spike/rolling_avg.npy'
    latency_path = "./logs/delta_sensitivity_spike/multi_ddqn_delta0p05_seed42_spike/adaptation_latency.npy"      # e.g., './logs/delta_sensitivity_spike/single_ddqn_delta0p05_seed42_spike/adaptation_latency.npy'
    
    if not os.path.exists(rolling_avg_path) or not os.path.exists(latency_path):
        print("Figure 3: Data files not found. Skipping.")
        return
    
    rolling_avg = np.load(rolling_avg_path)
    latency = np.load(latency_path)[0] if latency_path else 0
    spike_episode = 250
    recovery_episode = spike_episode + int(latency)
    flow_low, flow_high = 220, 470  # Change to 220, 470 for multi
    
    episodes = np.arange(len(rolling_avg))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(episodes, rolling_avg, linewidth=2, color='#2c3e50', label='Rolling Average')
    ax.axhspan(flow_low, flow_high, alpha=0.2, color='green', label='Flow Zone')
    ax.axvline(x=spike_episode, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Spike')
    ax.axvline(x=recovery_episode, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='Recovery')
    ax.annotate(f'Latency = {latency} eps', xy=((spike_episode+recovery_episode)/2, 1500),
                xytext=((spike_episode+recovery_episode)/2, 1700),
                arrowprops=dict(arrowstyle='->'), ha='center')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Rolling Average Reward', fontsize=12)
    ax.set_title('Figure 3 — Reward Stability Under Dynamic Difficulty', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./logs/figures/figure3_multi_reward_stability.png', dpi=300)
    plt.close()
    print("Figure 3 saved.")

# =============================================================================
# FIGURE 4 — Adaptation Latency
# =============================================================================
def figure4_adaptation_latency():
    """
    Needs: episode_rewards.npy, rolling_avg.npy, adaptation_latency.npy, spike episode.
    Placeholder paths.
    """
    # --- FILL THESE FILE PATHS ---
    rewards_path = "./logs/delta_sensitivity_spike/multi_ddqn_delta0p05_seed42_spike/episode_rewards.npy"      # e.g., './logs/delta_sensitivity_spike/multi_ddqn_delta0p05_seed42_spike/episode_rewards.npy'
    rolling_path = "./logs/delta_sensitivity_spike/multi_ddqn_delta0p05_seed42_spike/rolling_avg.npy"      # e.g., './logs/delta_sensitivity_spike/multi_ddqn_delta0p05_seed42_spike/rolling_avg.npy'
    latency_path = "./logs/delta_sensitivity_spike/multi_ddqn_delta0p05_seed42_spike/adaptation_latency.npy"      # e.g., './logs/delta_sensitivity_spike/multi_ddqn_delta0p05_seed42_spike/adaptation_latency.npy'
    
    if not all(os.path.exists(p) for p in [rewards_path, rolling_path, latency_path]):
        print("Figure 4: Data files not found. Skipping.")
        return
    
    rewards = np.load(rewards_path)
    rolling_avg = np.load(rolling_path)
    latency = np.load(latency_path)[0]
    spike_episode = 250
    recovery_episode = spike_episode + int(latency)
    flow_low, flow_high = 220, 470
    
    episodes = np.arange(len(rewards))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(episodes, rewards, s=5, alpha=0.4, color='#3498db', label='Episode Reward')
    ax.plot(episodes, rolling_avg, linewidth=2, color='#2c3e50', label='Rolling Average')
    ax.axhspan(flow_low, flow_high, alpha=0.15, color='green')
    ax.axvline(x=spike_episode, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Spike')
    ax.axvline(x=recovery_episode, color='blue', linestyle='--', linewidth=2, alpha=0.8, label='Recovery')
    ax.annotate(f'Latency = {latency} eps', xy=((spike_episode+recovery_episode)/2, 1600),
                xytext=((spike_episode+recovery_episode)/2, 1850),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2), ha='center')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Figure 4 — Adaptation Latency', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-50, 2200)
    plt.tight_layout()
    plt.savefig('./logs/figures/figure4_adaptation_latency_multi.png', dpi=300)
    plt.close()
    print("Figure 4 saved.")

# =============================================================================
# FIGURE 5 — Delta Sensitivity Curves
# =============================================================================
def figure5_delta_sensitivity():
    """
    Needs: aggregated CSV with columns: environment, delta, fzc_mean, oscillation_freq_mean.
    Placeholder: path to delta_sensitivity_aggregated.csv.
    """
    # --- FILL THIS FILE PATH ---
    agg_csv = "./logs/delta_sensitivity/delta_sensitivity_aggregated.csv"  # e.g., './logs/delta_sensitivity/delta_sensitivity_aggregated.csv'
    
    if not os.path.exists(agg_csv):
        print("Figure 5: Data file not found. Skipping.")
        return
    
    df = pd.read_csv(agg_csv)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, env in enumerate(['single', 'multi']):
        env_df = df[df['environment'] == env]
        ax = axes[i]
        ax.plot(env_df['delta'], env_df['fzc_mean'], marker='o', linewidth=2, color='#2c3e50', label='FZC')
        ax2 = ax.twinx()
        ax2.plot(env_df['delta'], env_df['oscillation_freq_mean'], marker='s', linewidth=2, color='#e74c3c', label='Oscillation')
        ax.set_xlabel('Delta (Δd)')
        ax.set_ylabel('FZC (%)')
        ax2.set_ylabel('Oscillation Frequency')
        ax.set_title(f'{env.upper()} Environment')
        ax.grid(True, alpha=0.3)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
    fig.suptitle('Figure 5 — Delta Sensitivity Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./logs/figures/figure5_delta_sensitivity.png', dpi=300)
    plt.close()
    print("Figure 5 saved.")

# =============================================================================
# FIGURE 6 — Boxplots Across Seeds
# =============================================================================
def figure6_boxplots():
    """
    Needs: raw seeds CSV for steady and spike.
    Placeholder paths.
    """
    # --- FILL THESE FILE PATHS ---
    steady_csv = "./logs/delta_sensitivity/delta_sensitivity_raw_seeds.csv"  # e.g., './logs/delta_sensitivity/delta_sensitivity_raw_seeds.csv'
    spike_csv  = "./logs/delta_sensitivity_spike/delta_sensitivity_raw_seeds_spike.csv"  # e.g., './logs/delta_sensitivity_spike/delta_sensitivity_raw_seeds_spike.csv'
    
    if not os.path.exists(steady_csv) or not os.path.exists(spike_csv):
        print("Figure 6: Data files not found. Skipping.")
        return
    
    df_steady = pd.read_csv(steady_csv)
    df_spike = pd.read_csv(spike_csv)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Mean Reward boxplot
    sns.boxplot(data=df_steady, x='environment', y='mean_reward', hue='delta', ax=axes[0])
    axes[0].set_title('Mean Reward')
    axes[0].legend(title='Delta', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # FZC boxplot
    sns.boxplot(data=df_steady, x='environment', y='fzc', hue='delta', ax=axes[1])
    axes[1].set_title('FZC (%)')
    axes[1].legend(title='Delta', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Latency boxplot (spike)
    sns.boxplot(data=df_spike, x='environment', y='adaptation_latency', hue='delta', ax=axes[2])
    axes[2].set_title('Adaptation Latency (eps)')
    axes[2].legend(title='Delta', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    fig.suptitle('Figure 6 — Boxplots Across Seeds', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./logs/figures/figure6_boxplots.png', dpi=300)
    plt.close()
    print("Figure 6 saved.")

# =============================================================================
# FIGURE 7 — Static vs Adaptive Difficulty Comparison
# =============================================================================
def figure7_static_vs_adaptive():
    """
    Static vs Adaptive Difficulty Comparison using ablation results.
    Only DDQN is shown because only DDQN was run in the ablation.
    """
    algorithms = ['DDQN']  # Only DDQN

    # Static data from No DDA ablation (Single environment)
    static_rewards = [988.4]
    static_fzc = [78.9]

    # Adaptive data from Full System (Table 3)
    adaptive_rewards = [1174.3]
    adaptive_fzc = [63.8]
    
    x = np.arange(len(algorithms))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].bar(x - width/2, static_rewards, width, label='Static (No DDA)', color='#e74c3c', alpha=0.7)
    axes[0].bar(x + width/2, adaptive_rewards, width, label='Adaptive (DDA)', color='#2ecc71', alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(algorithms)
    axes[0].set_ylabel('Mean Reward')
    axes[0].set_title('Mean Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(x - width/2, static_fzc, width, label='Static (No DDA)', color='#e74c3c', alpha=0.7)
    axes[1].bar(x + width/2, adaptive_fzc, width, label='Adaptive (DDA)', color='#2ecc71', alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(algorithms)
    axes[1].set_ylabel('FZC (%)')
    axes[1].set_title('Flow Zone Consistency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle('Figure 7 — Static vs Adaptive Difficulty Comparison (DDQN)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./logs/figures/figure7_static_vs_adaptive.png', dpi=300)
    plt.close()
    print("Figure 7 saved.")

# =============================================================================
# FIGURE 8 — Difficulty Oscillation Dynamics
# =============================================================================
def figure8_oscillation():
    """
    Needs: speed_history.npy for several deltas.
    Placeholder paths for different deltas.
    """
    # --- FILL THESE PATHS (one per delta) ---
    speed_paths = {
        0.05: "./logs/delta_sensitivity/single_ddqn_delta0p05_seed42/speed_history.npy",  # './logs/delta_sensitivity/single_ddqn_delta0p05_seed42/speed_history.npy'
        0.10: "./logs/delta_sensitivity/single_ddqn_delta0p1_seed42/speed_history.npy",  # './logs/delta_sensitivity/single_ddqn_delta0p1_seed42/speed_history.npy'
        0.20: "./logs/delta_sensitivity/single_ddqn_delta0p2_seed42/speed_history.npy",
        0.30: "./logs/delta_sensitivity/single_ddqn_delta0p3_seed42/speed_history.npy",
        0.50: "./logs/delta_sensitivity/single_ddqn_delta0p5_seed42/speed_history.npy",
        1.00: "./logs/delta_sensitivity/single_ddqn_delta1p0_seed42/speed_history.npy",
    }
    
    if not any(os.path.exists(p) for p in speed_paths.values() if p != "PLACEHOLDER"):
        print("Figure 8: No speed data found. Skipping.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f1c40f', '#9b59b6', '#1abc9c']
    for i, (delta, path) in enumerate(speed_paths.items()):
        if os.path.exists(path) and path != "PLACEHOLDER":
            speed = np.load(path)
            ax.plot(speed, color=colors[i % len(colors)], label=f'Δ={delta}', alpha=0.7)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Enemy Speed', fontsize=12)
    ax.set_title('Figure 8 — Difficulty Oscillation Dynamics', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./logs/figures/figure8_oscillation.png', dpi=300)
    plt.close()
    print("Figure 8 saved.")

# =============================================================================
# FIGURE 9 — Correlation Scatter Plot
# =============================================================================
def figure9_correlation():
    """
    Needs: episode_rewards.npy and speed_history.npy for a single run.
    Placeholder paths.
    """
    # --- FILL THESE PATHS ---
    rewards_path = "./logs/delta_sensitivity_spike/single_ddqn_delta0p05_seed42_spike/episode_rewards.npy"  # e.g., './logs/delta_sensitivity_spike/single_ddqn_delta0p05_seed42_spike/episode_rewards.npy'
    speed_path = "./logs/delta_sensitivity_spike/single_ddqn_delta0p05_seed42_spike/speed_history.npy"    # e.g., './logs/delta_sensitivity_spike/single_ddqn_delta0p05_seed42_spike/speed_history.npy'
    
    if not os.path.exists(rewards_path) or not os.path.exists(speed_path):
        print("Figure 9: Data files not found. Skipping.")
        return
    
    rewards = np.load(rewards_path)
    speed = np.load(speed_path)
    
    # Compute reward stability (std of rewards) per speed? Or we can use per-episode correlation.
    # For simplicity, scatter rewards vs speed.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(speed, rewards, alpha=0.5, s=10, color='#2c3e50')
    ax.set_xlabel('Enemy Speed', fontsize=12)
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title('Figure 9 — Correlation: Difficulty vs Reward', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./logs/figures/figure9_correlation.png', dpi=300)
    plt.close()
    print("Figure 9 saved.")

# =============================================================================
# FIGURE 10 — Generalization Performance Across Environments
# =============================================================================
def figure10_generalization():
    """
    Needs: data from Table 3 for DDQN and PPO in single and multi.
    Can be hardcoded or loaded.
    """
    # --- FILL THESE VALUES (from Table 3) ---
    environments = ['Single', 'Multi']
    ddqn_rewards = [1174.3, 337.7]   # Mean rewards
    ppo_rewards = [1005.1, 352.7]    # Mean rewards
    # Optionally include FZC
    ddqn_fzc = [63.8, 85.6]
    ppo_fzc = [72.9, 72.3]
    
    x = np.arange(len(environments))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].bar(x - width/2, ddqn_rewards, width, label='DDQN', color='#2c3e50')
    axes[0].bar(x + width/2, ppo_rewards, width, label='PPO', color='#3498db')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(environments)
    axes[0].set_ylabel('Mean Reward')
    axes[0].set_title('Mean Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(x - width/2, ddqn_fzc, width, label='DDQN', color='#2c3e50')
    axes[1].bar(x + width/2, ppo_fzc, width, label='PPO', color='#3498db')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(environments)
    axes[1].set_ylabel('FZC (%)')
    axes[1].set_title('Flow Zone Consistency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle('Figure 10 — Generalization Performance Across Environments', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./logs/figures/figure10_generalization.png', dpi=300)
    plt.close()
    print("Figure 10 saved.")

# =============================================================================
# MAIN — Generate all figures
# =============================================================================
def main():
    print("Generating figures...\n")
    # figure2_training_curves()
    # figure3_reward_stability()
    figure4_adaptation_latency()
    # figure5_delta_sensitivity()
    # figure6_boxplots()
    # figure7_static_vs_adaptive()
    # figure8_oscillation()
    # figure9_correlation()
    # figure10_generalization()
    print("\nAll figures generated. Check ./logs/figures/")

if __name__ == "__main__":
    main()