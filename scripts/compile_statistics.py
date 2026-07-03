import os
import numpy as np
import scipy.stats as stats

data_dir = "./logs/multi_seed_eval/"
deltas = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]
seeds = [42, 105, 2026, 777, 999]

# Dictionary to hold the statistical arrays
results = {}

print(f"{'Delta':<6} | {'Mean Reward (±SD)':<22} | {'Mean FZC (±SD)':<18} | {'95% CI (FZC)':<16} | {'p-value vs 0.05'}")
print("-" * 85)

# We evaluate Delta = 0.05 as our base control configuration
control_fzc_scores = []

for delta in deltas:
    rewards_per_seed = []
    fzc_per_seed = []
    
    for seed in seeds:
        reward_file = os.path.join(data_dir, f"rewards_delta_{delta}_seed_{seed}.npy")
        if os.path.exists(reward_file):
            rewards = np.load(reward_file)
            mean_reward = np.mean(rewards)
            
            # Flow zone constraints matching your controller logic (600.0 to 1200.0)
            fzc = np.mean((rewards >= 600.0) & (rewards <= 1200.0))
            
            rewards_per_seed.append(mean_reward)
            fzc_per_seed.append(fzc)
            
    if len(fzc_per_seed) == 0:
        continue
        
    if delta == 0.05:
        control_fzc_scores = fzc_per_seed

    # Compute Core Metrics
    m_reward, sd_reward = np.mean(rewards_per_seed), np.std(rewards_per_seed)
    m_fzc, sd_fzc = np.mean(fzc_per_seed), np.std(fzc_per_seed)
    
    # 95% Confidence Interval Calculation
    n = len(seeds)
    stderr_fzc = stats.sem(fzc_per_seed)
    ci_fzc = stderr_fzc * stats.t.ppf((1 + 0.95) / 2., n - 1) if n > 1 else 0.0

    # Mann-Whitney U test vs delta 0.05 control
    if delta == 0.05:
        p_val_str = "Control"
    else:
        _, p_val = stats.mannwhitneyu(control_fzc_scores, fzc_per_seed, alternative='two-sided')
        p_val_str = f"{p_val:.4f}"

    print(f"{delta:<6} | {m_reward:>6.1f} (±{sd_reward:>4.1f})    | {m_fzc:>5.1%} (±{sd_fzc:>4.1%})  | [{m_fzc-ci_fzc:.1%}, {m_fzc+ci_fzc:.1%}] | {p_val_str}")