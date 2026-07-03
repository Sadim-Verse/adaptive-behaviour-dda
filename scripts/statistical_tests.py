# scripts/statistical_tests.py
"""
STATISTICAL SIGNIFICANCE TESTS FOR TABLE 4

Computes pairwise comparisons between algorithms on:
    - Single Environment (FZC, Mean Reward, Adaptation Latency)
    - Multi Environment (FZC, Mean Reward, Adaptation Latency)

Tests used:
    - Mann-Whitney U (non-parametric, for FZC and Reward Stability)
    - Welch t-test (parametric, handles unequal variances, for Latency)

Data: 5 seeds per algorithm (42, 105, 777, 999, 2026)
Significance threshold: p < 0.05
"""

import os
import numpy as np
from scipy import stats
from itertools import combinations

# ── Configuration ────────────────────────────────────────────────────────── #

BASE_LOG_DIR = "./logs/experiments"
ALGOS = ["dqn", "ddqn", "ppo"]
SEEDS = [42, 105, 777, 999, 2026]
ENVS = ["single", "multi"]
WINDOW_SIZE = 10

# Flow zone bounds
FLOW_ZONES = {
    "single": (600.0, 1200.0),
    "multi": (220.0, 470.0)
}

# Algorithm folder mapping
ALGO_FOLDERS = {
    "dqn": "dqn",
    "ddqn": "ddqn",
    "ppo": "ppo"
}

# ── Helper Functions ─────────────────────────────────────────────────────── #

def load_metrics(algo, env):
    """Load per-seed metrics for one algorithm and environment."""
    rewards_all = []
    fzc_all = []
    latency_all = []

    flow_low, flow_high = FLOW_ZONES[env]
    algo_folder = ALGO_FOLDERS[algo]

    for seed in SEEDS:
        run_folder = f"{env}_{algo}_delta0p05_seed{seed}"
        folder_path = os.path.join(BASE_LOG_DIR, algo_folder, run_folder)

        rewards_file = os.path.join(folder_path, "episode_rewards.npy")
        rolling_file = os.path.join(folder_path, "rolling_avg.npy")

        if not (os.path.exists(rewards_file) and os.path.exists(rolling_file)):
            print(f"  ⚠️ Missing files for {algo.upper()} {env} seed {seed}")
            continue

        rewards = np.load(rewards_file)
        rolling_avg = np.load(rolling_file)

        # Mean episode reward
        mean_rew = float(np.mean(rewards))

        # FZC — proportion of rolling-avg episodes inside Flow Zone
        valid_avgs = rolling_avg[WINDOW_SIZE - 1:]
        fzc = float(np.mean((valid_avgs >= flow_low) & (valid_avgs <= flow_high))) * 100.0

        # Adaptation Latency — first episode where rolling avg enters flow zone
        latency = len(rolling_avg)
        for idx, avg in enumerate(rolling_avg):
            if flow_low <= avg <= flow_high:
                latency = idx + 1
                break

        rewards_all.append(mean_rew)
        fzc_all.append(fzc)
        latency_all.append(float(latency))

    return {
        "mean_reward": np.array(rewards_all),
        "fzc": np.array(fzc_all),
        "latency": np.array(latency_all),
    }


def load_latency_spike(algo, env):
    """Load Adaptation Latency from SPIKE mode runs."""
    latency_all = []
    flow_low, flow_high = FLOW_ZONES[env]
    algo_folder = ALGO_FOLDERS[algo]

    for seed in SEEDS:
        run_folder = f"{env}_{algo}_delta0p05_seed{seed}_spike"
        folder_path = os.path.join(BASE_LOG_DIR, algo_folder, run_folder)

        latency_file = os.path.join(folder_path, "adaptation_latency.npy")

        if not os.path.exists(latency_file):
            print(f"  ⚠️ Missing latency file for {algo.upper()} {env} seed {seed} (spike mode)")
            continue

        latency = np.load(latency_file)
        if len(latency) > 0:
            latency_all.append(float(latency[0]))

    return np.array(latency_all)


def run_mannwhitney(a_data, b_data):
    """Run Mann-Whitney U test."""
    if len(a_data) < 2 or len(b_data) < 2:
        return None
    stat, p = stats.mannwhitneyu(a_data, b_data, alternative="two-sided")
    return stat, p


def run_welch_ttest(a_data, b_data):
    """Run Welch t-test (unequal variance)."""
    if len(a_data) < 2 or len(b_data) < 2:
        return None
    stat, p = stats.ttest_ind(a_data, b_data, equal_var=False)
    return stat, p


def sig_stars(p):
    """Convert p-value to significance stars."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ── Main ─────────────────────────────────────────────────────────────────── #

print("=" * 90)
print("TABLE 4 — STATISTICAL SIGNIFICANCE TESTS")
print("=" * 90)
print(f"  Algorithms: {', '.join([a.upper() for a in ALGOS])}")
print(f"  Seeds: {SEEDS}")
print(f"  Environments: {', '.join([e.upper() for e in ENVS])}")
print(f"  Significance threshold: p < 0.05")
print("=" * 90)

# Store results for formatting
table_rows = []

for env in ENVS:
    print(f"\n{'─'*90}")
    print(f"ENVIRONMENT: {env.upper()}")
    print(f"{'─'*90}")

    # Load data for this environment
    data = {}
    for algo in ALGOS:
        data[algo] = load_metrics(algo, env)

    # Load spike latency data
    spike_latency = {}
    for algo in ALGOS:
        spike_latency[algo] = load_latency_spike(algo, env)

    # Descriptive summary
    print(f"\nDescriptive Summary ({env.upper()}):")
    print(f"{'Algorithm':<10} | {'Mean Reward':<15} | {'FZC (%)':<12} | {'Latency (eps)':<15}")
    print("-" * 60)
    for algo in ALGOS:
        d = data[algo]
        lat_mean = np.mean(spike_latency[algo]) if len(spike_latency[algo]) > 0 else float('nan')
        lat_std = np.std(spike_latency[algo]) if len(spike_latency[algo]) > 0 else float('nan')
        print(f"{algo.upper():<10} | "
              f"{np.mean(d['mean_reward']):>6.1f} ± {np.std(d['mean_reward']):<5.1f} | "
              f"{np.mean(d['fzc']):>5.1f}% ± {np.std(d['fzc']):<4.1f}% | "
              f"{lat_mean:>5.1f} ± {lat_std:<5.1f}")

    # Pairwise comparisons
    print(f"\nPairwise Comparisons ({env.upper()}):")
    print("-" * 90)
    print(f"{'Comparison':<16} | {'Metric':<22} | {'Test':<18} | {'p-value':>10} | {'Sig':>5} | {'Significant?'}")
    print("-" * 90)

    pairs = list(combinations(ALGOS, 2))

    for algo_a, algo_b in pairs:
        # FZC (Mann-Whitney U)
        stat, p = run_mannwhitney(data[algo_a]["fzc"], data[algo_b]["fzc"])
        if stat is not None:
            sig = p < 0.05
            print(f"{algo_a.upper()} vs {algo_b.upper():<8} | FZC{' ':<19} | Mann-Whitney U{' ':<6} | "
                  f"{p:>10.4f} | {sig_stars(p):>5} | {'YES' if sig else 'NO'}")

        # Mean Reward (Welch t-test)
        stat, p = run_welch_ttest(data[algo_a]["mean_reward"], data[algo_b]["mean_reward"])
        if stat is not None:
            sig = p < 0.05
            print(f"{algo_a.upper()} vs {algo_b.upper():<8} | Mean Reward{' ':<12} | Welch t-test{' ':<8} | "
                  f"{p:>10.4f} | {sig_stars(p):>5} | {'YES' if sig else 'NO'}")

        # Adaptation Latency (Spike) (Welch t-test)
        a_lat = spike_latency[algo_a]
        b_lat = spike_latency[algo_b]
        if len(a_lat) > 0 and len(b_lat) > 0:
            stat, p = run_welch_ttest(a_lat, b_lat)
            if stat is not None:
                sig = p < 0.05
                print(f"{algo_a.upper()} vs {algo_b.upper():<8} | Adaptation Latency{' ':<5} | Welch t-test{' ':<8} | "
                      f"{p:>10.4f} | {sig_stars(p):>5} | {'YES' if sig else 'NO'}")

    print("-" * 90)

print("\n" + "=" * 90)
print("LEGEND:")
print("  *  p < 0.05  |  **  p < 0.01  |  ***  p < 0.001  |  ns = not significant")
print("  Mann-Whitney U used for FZC (non-parametric, n=5)")
print("  Welch t-test used for Mean Reward and Latency (parametric, unequal variance)")
print("  Adaptation Latency measured from SPIKE mode (recovery from difficulty shock)")
print("=" * 90)

# ── Formatted Table 4 Output ────────────────────────────────────────────── #

print("\n\n" + "=" * 90)
print("TABLE 4 — FORMATTED FOR THESIS")
print("=" * 90)
print(f"{'Comparison':<16} | {'Metric':<22} | {'Test':<18} | {'p-value':>10} | {'Significant?'}")
print("-" * 90)

for env in ENVS:
    print(f"\n{env.upper()} ENVIRONMENT:")
    data = {}
    for algo in ALGOS:
        data[algo] = load_metrics(algo, env)
    spike_latency = {}
    for algo in ALGOS:
        spike_latency[algo] = load_latency_spike(algo, env)

    pairs = list(combinations(ALGOS, 2))
    for algo_a, algo_b in pairs:
        # FZC
        stat, p = run_mannwhitney(data[algo_a]["fzc"], data[algo_b]["fzc"])
        if stat is not None:
            sig = p < 0.05
            comparison = f"{algo_a.upper()} vs {algo_b.upper()}"
            print(f"{comparison:<16} | FZC{' ':<19} | Mann-Whitney U{' ':<6} | {p:>10.4f} | {'Yes' if sig else 'No'}")

        # Mean Reward
        stat, p = run_welch_ttest(data[algo_a]["mean_reward"], data[algo_b]["mean_reward"])
        if stat is not None:
            sig = p < 0.05
            comparison = f"{algo_a.upper()} vs {algo_b.upper()}"
            print(f"{comparison:<16} | Mean Reward{' ':<12} | Welch t-test{' ':<8} | {p:>10.4f} | {'Yes' if sig else 'No'}")

        # Adaptation Latency (Spike)
        a_lat = spike_latency[algo_a]
        b_lat = spike_latency[algo_b]
        if len(a_lat) > 0 and len(b_lat) > 0:
            stat, p = run_welch_ttest(a_lat, b_lat)
            if stat is not None:
                sig = p < 0.05
                comparison = f"{algo_a.upper()} vs {algo_b.upper()}"
                print(f"{comparison:<16} | Adaptation Latency{' ':<5} | Welch t-test{' ':<8} | {p:>10.4f} | {'Yes' if sig else 'No'}")

print("=" * 90)