# scripts/statistical_tests.py
"""
STATISTICAL SIGNIFICANCE TESTS FOR TABLE 4

Computes pairwise comparisons between algorithms on:
    - Single, Multi, Procedural (ObstacleMiniGrid) environments
    - FZC (Mann-Whitney U), Mean Reward (Welch t-test), Adaptation Latency (Welch t-test)

Data: 5 seeds per algorithm (42, 105, 777, 999, 2026)
Significance threshold: p < 0.05

FIXES APPLIED IN THIS VERSION:
1. Path construction no longer adds an extra "ddqn"/"dqn"/"ppo" subfolder
   for Single/Multi environments — the data is in the flat structure directly
   under ./logs/experiments/ (same as the ablation script).
2. Adaptation Latency now uses sample standard deviation (ddof=1) consistently
   with the established convention in run_delta_sensitivity.py.
3. FLOW_ZONES imported from training.train_multi_env (single source of truth).
4. Procedural (obstacle_minigrid) included with its own base log directory.
5. Sentinel values (>= 500) excluded from latency comparisons.
"""

import os
import argparse
import numpy as np
from scipy import stats
from itertools import combinations

# ── Configuration ────────────────────────────────────────────────────────── #

try:
    from training.train_multi_env import FLOW_ZONES as _FLOW_ZONES_RAW
    FLOW_ZONES = {env: bounds["default"] for env, bounds in _FLOW_ZONES_RAW.items()}
except ImportError:
    print("WARNING: could not import FLOW_ZONES from training.train_multi_env "
          "(run this as `python -m scripts.statistical_tests` from the project "
          "root). Falling back to last-confirmed printed values.")
    FLOW_ZONES = {
        "single": (296.0, 765.0),
        "multi": (245.6, 470.1),
        "obstacle_minigrid": (-74.5, -23.7),
    }

ALGOS = ["dqn", "ddqn", "ppo"]
SEEDS = [42, 105, 777, 999, 2026]
ENVS = ["single", "multi", "obstacle_minigrid"]
WINDOW_SIZE = 10
N_EPISODES_SPIKE_RUN = 500  # non-recovery sentinel cutoff

BASE_LOG_DIRS = {
    "single": "./logs/experiments",
    "multi": "./logs/experiments",
    "obstacle_minigrid": "./logs/procedural_env_v2",
}


def run_folder_name(env, algo, seed, spike=False):
    suffix = "_spike" if spike else ""
    return f"{env}_{algo}_delta0p05_seed{seed}{suffix}"


def run_folder_path(env, algo, seed, spike=False):
    """
    FLAT STRUCTURE — no extra per-algo subfolder for any environment.
    This matches where run_all_evaluations.py saves the data.
    """
    base = BASE_LOG_DIRS[env]
    folder = run_folder_name(env, algo, seed, spike)
    return os.path.join(base, folder)


# ── Helper Functions ─────────────────────────────────────────────────────── #

def load_metrics(algo, env):
    """Load per-seed steady-mode metrics (mean reward, FZC) for one algorithm/environment."""
    rewards_all = []
    fzc_all = []
    flow_low, flow_high = FLOW_ZONES[env]

    for seed in SEEDS:
        folder_path = run_folder_path(env, algo, seed, spike=False)
        rewards_file = os.path.join(folder_path, "episode_rewards.npy")
        rolling_file = os.path.join(folder_path, "rolling_avg.npy")

        if not (os.path.exists(rewards_file) and os.path.exists(rolling_file)):
            print(f"  WARNING: missing files for {algo.upper()} {env} seed {seed}: {folder_path}")
            continue

        rewards = np.load(rewards_file)
        rolling_avg = np.load(rolling_file)

        mean_rew = float(np.mean(rewards))
        valid_avgs = rolling_avg[WINDOW_SIZE - 1:]
        fzc = float(np.mean((valid_avgs >= flow_low) & (valid_avgs <= flow_high))) * 100.0

        rewards_all.append(mean_rew)
        fzc_all.append(fzc)

    return {
        "mean_reward": np.array(rewards_all),
        "fzc": np.array(fzc_all),
    }


def load_latency_spike(algo, env, n_episodes=N_EPISODES_SPIKE_RUN):
    """
    Load Adaptation Latency from SPIKE mode runs, split into recovered vs
    censored (sentinel) seeds.
    Returns (recovered_latencies, n_total, recovery_rate).
    """
    all_latencies = []
    for seed in SEEDS:
        folder_path = run_folder_path(env, algo, seed, spike=True)
        latency_file = os.path.join(folder_path, "adaptation_latency.npy")
        if not os.path.exists(latency_file):
            print(f"  WARNING: missing latency file for {algo.upper()} {env} seed {seed} (spike): {folder_path}")
            continue
        latency = np.load(latency_file)
        if len(latency) > 0:
            all_latencies.append(float(np.asarray(latency).flatten()[0]))

    all_latencies = np.array(all_latencies)
    n_total = len(all_latencies)
    recovered = all_latencies[all_latencies < n_episodes] if n_total else all_latencies
    recovery_rate = (len(recovered) / n_total) if n_total else float("nan")
    return recovered, n_total, recovery_rate


def run_mannwhitney(a_data, b_data):
    if len(a_data) < 2 or len(b_data) < 2:
        return None
    return stats.mannwhitneyu(a_data, b_data, alternative="two-sided")


def run_welch_ttest(a_data, b_data):
    if len(a_data) < 2 or len(b_data) < 2:
        return None
    return stats.ttest_ind(a_data, b_data, equal_var=False)


def sig_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ── Main ─────────────────────────────────────────────────────────────────── #

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--envs", nargs="+", default=ENVS, choices=ENVS)
    args = ap.parse_args()

    print("=" * 90)
    print("TABLE 4 -- STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 90)
    print(f"  Algorithms: {', '.join(a.upper() for a in ALGOS)}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Environments: {', '.join(e.upper() for e in args.envs)}")
    print(f"  Flow Zones (imported): {FLOW_ZONES}")
    print(f"  Significance threshold: p < 0.05")
    print("=" * 90)

    for env in args.envs:
        print(f"\n{'-'*90}")
        print(f"ENVIRONMENT: {env.upper()}")
        print(f"{'-'*90}")

        data = {algo: load_metrics(algo, env) for algo in ALGOS}
        spike_latency = {}
        for algo in ALGOS:
            recovered, n_total, rate = load_latency_spike(algo, env)
            spike_latency[algo] = recovered
            if n_total:
                lat_str = (f"{np.mean(recovered):.1f} +/- {np.std(recovered, ddof=1):.1f} eps "
                           f"({len(recovered)}/{n_total} recovered)")
                print(f"  {algo.upper():<5} recovery rate: {rate:.0%} | latency: {lat_str}")
            else:
                print(f"  {algo.upper():<5} no data")

        print(f"\nDescriptive Summary ({env.upper()}):")
        print(f"{'Algorithm':<10} | {'Mean Reward':<15} | {'FZC (%)':<12}")
        print("-" * 45)
        for algo in ALGOS:
            d = data[algo]
            if len(d["mean_reward"]) == 0:
                print(f"{algo.upper():<10} | NO DATA")
                continue
            print(f"{algo.upper():<10} | "
                  f"{np.mean(d['mean_reward']):>6.1f} +/- {np.std(d['mean_reward']):<5.1f} | "
                  f"{np.mean(d['fzc']):>5.1f}% +/- {np.std(d['fzc']):<4.1f}%")

        print(f"\nPairwise Comparisons ({env.upper()}):")
        print("-" * 90)
        print(f"{'Comparison':<16} | {'Metric':<20} | {'Test':<16} | {'p-value':>10} | {'Sig':>5} | Note")
        print("-" * 90)

        for algo_a, algo_b in combinations(ALGOS, 2):
            comp = f"{algo_a.upper()} vs {algo_b.upper()}"

            res = run_mannwhitney(data[algo_a]["fzc"], data[algo_b]["fzc"])
            if res is not None:
                p = res.pvalue
                print(f"{comp:<16} | {'FZC':<20} | {'Mann-Whitney U':<16} | {p:>10.4f} | {sig_stars(p):>5} |")

            res = run_welch_ttest(data[algo_a]["mean_reward"], data[algo_b]["mean_reward"])
            if res is not None:
                p = res.pvalue
                print(f"{comp:<16} | {'Mean Reward':<20} | {'Welch t-test':<16} | {p:>10.4f} | {sig_stars(p):>5} |")

            a_lat, b_lat = spike_latency[algo_a], spike_latency[algo_b]
            res = run_welch_ttest(a_lat, b_lat)
            if res is not None:
                p = res.pvalue
                note = "" if (len(a_lat) == len(SEEDS) and len(b_lat) == len(SEEDS)) else \
                       f"partial recovery ({len(a_lat)}/{len(SEEDS)} vs {len(b_lat)}/{len(SEEDS)})"
                print(f"{comp:<16} | {'Adaptation Latency':<20} | {'Welch t-test':<16} | {p:>10.4f} | {sig_stars(p):>5} | {note}")
            else:
                print(f"{comp:<16} | {'Adaptation Latency':<20} | {'Welch t-test':<16} | {'SKIPPED':>10} | {'':>5} | "
                      f"fewer than 2 recovered seeds in at least one group")

        print("-" * 90)

    print("\nLEGEND: *  p<0.05 | **  p<0.01 | ***  p<0.001 | ns = not significant")
    print("Mann-Whitney U for FZC (non-parametric, n=5). Welch t-test for Mean Reward and")
    print("Latency (parametric, unequal variance). Latency comparisons exclude non-recovery")
    print("sentinel values (Priority 6) -- recovery rate is reported alongside each result.")


if __name__ == "__main__":
    main()