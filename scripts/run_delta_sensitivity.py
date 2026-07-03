# scripts/run_delta_sensitivity.py
"""
Run delta sensitivity analysis by calling train_multi_env.py for each configuration.
- 6 delta values: 0.05, 0.10, 0.20, 0.30, 0.50, 1.00
- 2 environments: single, multi
- 5 seeds: 42, 105, 777, 999, 2026
- 1 mode: SPIKE (at episode 250)

Total runs: 6 deltas × 2 envs × 5 seeds = 60 runs (~3 hours)
Results saved to: ./logs/delta_sensitivity_spike/
Each run saved to: {env}_ddqn_delta{delta}_seed{seed}_spike/
"""

import os
import sys
import subprocess
import time
import numpy as np
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────── #

DELTAS = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]
ENVIRONMENTS = ["single", "multi"]
SEEDS = [42, 105, 777, 999, 2026]
MAX_SPEED = 9.0
SPIKE_EPISODE = 250

BASE_LOG_DIR = "./logs/delta_sensitivity_spike"

# Flow zone bounds
FLOW_ZONES = {
    "single": (600.0, 1200.0),
    "multi": (220.0, 470.0)
}

# Paths to trained models
MODEL_PATHS = {
    "single": "./logs/experiments/ddqn/single_ddqn_delta0p05_seed42/best_model.zip",
    "multi": "./logs/experiments/ddqn/multi_ddqn_delta0p05_seed42/best_model.zip"
}

# ── Helper Functions ─────────────────────────────────────────────────────── #

def compute_oscillation_frequency(speed_history):
    if len(speed_history) < 3:
        return 0.0
    diffs = np.diff(speed_history)
    direction_changes = 0
    for i in range(1, len(diffs)):
        if diffs[i] * diffs[i-1] < 0:
            direction_changes += 1
    return direction_changes / len(speed_history)

def compute_survival_time(rewards):
    survival_times = []
    for r in rewards:
        if r >= 2000:
            survival_times.append(2000)
        elif r <= -100:
            survival_times.append(0)
        else:
            survival_times.append(r + 100)
    return np.mean(survival_times)

def compute_fzc(rolling_avgs, flow_low, flow_high, window_size=10):
    if len(rolling_avgs) < window_size:
        return 0.0
    valid_avgs = rolling_avgs[window_size - 1:]
    in_zone = np.sum((valid_avgs >= flow_low) & (valid_avgs <= flow_high))
    return (in_zone / len(valid_avgs)) * 100.0

def load_adaptation_latency(folder_path):
    """Load adaptation latency from .npy file (if it exists)."""
    latency_file = os.path.join(folder_path, "adaptation_latency.npy")
    if os.path.exists(latency_file):
        latency = np.load(latency_file)
        if len(latency) > 0:
            return float(latency[0])
    return None

# ── Main ─────────────────────────────────────────────────────────────────── #

def main():
    print("=" * 80)
    print("DELTA SENSITIVITY ANALYSIS (MULTI-SEED) - SPIKE MODE")
    print(f"  Algorithm: DDQN")
    print(f"  Spike at episode: {SPIKE_EPISODE}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Deltas: {DELTAS}")
    print(f"  Environments: {ENVIRONMENTS}")
    print(f"  Total runs: {len(DELTAS) * len(ENVIRONMENTS) * len(SEEDS)}")
    print("=" * 80)

    os.makedirs(BASE_LOG_DIR, exist_ok=True)

    # Store raw results per seed
    all_results = []

    for env in ENVIRONMENTS:
        print(f"\n{'─'*80}")
        print(f"ENVIRONMENT: {env.upper()}")
        print(f"{'─'*80}")

        model_path = MODEL_PATHS[env]
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found: {model_path}")
            continue

        for delta in DELTAS:
            print(f"\n  🔸 Delta = {delta}")
            delta_str = str(delta).replace('.', 'p')

            for seed in SEEDS:
                # ── Build folder name with _spike suffix ───────────────── #
                run_folder = f"{env}_ddqn_delta{delta_str}_seed{seed}_spike"
                log_dir = os.path.join(BASE_LOG_DIR, run_folder)
                os.makedirs(log_dir, exist_ok=True)

                # Check if data already exists
                rewards_path = os.path.join(log_dir, "episode_rewards.npy")
                if os.path.exists(rewards_path):
                    print(f"      Seed {seed}: ✅ Already exists — skipping")
                    rewards = np.load(rewards_path)
                    rolling_avgs = np.load(os.path.join(log_dir, "rolling_avg.npy"))
                    speed_history = np.load(os.path.join(log_dir, "speed_history.npy"))
                else:
                    # ── Run the evaluation with SPIKE ───────────────────── #
                    cmd = [
                        sys.executable, "-m", "training.train_multi_env",
                        "--algo", "ddqn",
                        "--env", env,
                        "--resume", model_path,
                        "--delta", str(delta),
                        "--seed", str(seed),
                        "--max_speed", str(MAX_SPEED),
                        "--spike", str(SPIKE_EPISODE),
                        "--output", BASE_LOG_DIR
                    ]

                    print(f"      Seed {seed}: Running...")
                    try:
                        subprocess.run(cmd, check=True, capture_output=False)
                    except subprocess.CalledProcessError as e:
                        print(f"      ❌ Seed {seed} failed with exit code {e.returncode}")
                        continue

                    try:
                        rewards = np.load(os.path.join(log_dir, "episode_rewards.npy"))
                        rolling_avgs = np.load(os.path.join(log_dir, "rolling_avg.npy"))
                        speed_history = np.load(os.path.join(log_dir, "speed_history.npy"))
                    except FileNotFoundError:
                        print(f"      ❌ Seed {seed} data not found after run")
                        continue

                # ── Compute metrics for this seed ───────────────────────── #
                flow_low, flow_high = FLOW_ZONES[env]
                mean_reward = np.mean(rewards)
                reward_std = np.std(rewards)
                fzc = compute_fzc(rolling_avgs, flow_low, flow_high)
                osc_freq = compute_oscillation_frequency(speed_history)
                survival = compute_survival_time(rewards)

                latency = load_adaptation_latency(log_dir)
                if latency is None:
                    latency = float('nan')

                print(f"          Mean Reward: {mean_reward:.1f} ± {reward_std:.1f}")
                print(f"          FZC: {fzc:.2f}%")
                print(f"          Oscillation: {osc_freq:.4f}")
                print(f"          Latency: {latency:.1f} eps" if not np.isnan(latency) else "          Latency: N/A")

                all_results.append({
                    "environment": env,
                    "delta": delta,
                    "seed": seed,
                    "mean_reward": mean_reward,
                    "reward_std": reward_std,
                    "fzc": fzc,
                    "oscillation_freq": osc_freq,
                    "survival_time": survival,
                    "adaptation_latency": latency,
                    "run_folder": run_folder
                })

    # ── Aggregate by Delta and Environment ──────────────────────────────── #

    if not all_results:
        print("\n⚠️ No results collected. All runs failed.")
        return

    df = pd.DataFrame(all_results)

    aggregated = df.groupby(["environment", "delta"]).agg({
        "mean_reward": ["mean", "std"],
        "reward_std": "mean",
        "fzc": ["mean", "std"],
        "oscillation_freq": ["mean", "std"],
        "survival_time": ["mean", "std"],
        "adaptation_latency": ["mean", "std"],
        "seed": "count"
    }).round(2)

    aggregated.columns = [
        "mean_reward", "reward_std_across_seeds",
        "reward_std_mean",
        "fzc_mean", "fzc_std",
        "oscillation_freq_mean", "oscillation_freq_std",
        "survival_time_mean", "survival_time_std",
        "latency_mean", "latency_std",
        "n_seeds"
    ]

    aggregated = aggregated.reset_index()

    # ── Save aggregated results ──────────────────────────────────────────── #

    agg_csv = os.path.join(BASE_LOG_DIR, "delta_sensitivity_aggregated_spike.csv")
    aggregated.to_csv(agg_csv, index=False)
    print(f"\n✅ Aggregated results saved to: {agg_csv}")

    print("\n" + "=" * 80)
    print("DELTA SENSITIVITY RESULTS (SPIKE MODE, 5 SEEDS)")
    print("=" * 80)

    for env in ENVIRONMENTS:
        print(f"\n📊 ENVIRONMENT: {env.upper()}")
        print("-" * 90)
        print(f"{'Delta':<8} | {'Mean Reward':<18} | {'FZC (%)':<12} | {'Oscillation':<12} | {'Latency (eps)':<12}")
        print("-" * 90)

        env_df = aggregated[aggregated["environment"] == env]
        for _, row in env_df.iterrows():
            lat_str = f"{row['latency_mean']:.1f} ± {row['latency_std']:.1f}" if not np.isnan(row['latency_mean']) else "N/A"
            print(f"{row['delta']:<8.2f} | "
                  f"{row['mean_reward']:>8.1f} ± {row['reward_std_across_seeds']:<5.1f} | "
                  f"{row['fzc_mean']:>8.1f}% ± {row['fzc_std']:<4.1f}% | "
                  f"{row['oscillation_freq_mean']:>8.4f} ± {row['oscillation_freq_std']:<.4f} | "
                  f"{lat_str}")
        print("-" * 90)

    # ── Save per-environment plotting CSVs ──────────────────────────────── #

    for env in ENVIRONMENTS:
        env_df = aggregated[aggregated["environment"] == env]
        if not env_df.empty:
            plot_df = env_df[[
                "delta",
                "mean_reward", "reward_std_across_seeds",
                "fzc_mean", "fzc_std",
                "oscillation_freq_mean", "oscillation_freq_std",
                "latency_mean", "latency_std"
            ]]
            plot_path = os.path.join(BASE_LOG_DIR, f"delta_sensitivity_{env}_for_plotting_spike.csv")
            plot_df.to_csv(plot_path, index=False)
            print(f"   Plotting data for {env}: {plot_path}")

    raw_csv = os.path.join(BASE_LOG_DIR, "delta_sensitivity_raw_seeds_spike.csv")
    df.to_csv(raw_csv, index=False)
    print(f"   Raw seed data: {raw_csv}")

    print("\n" + "=" * 80)
    print("✅ Spike-mode delta sensitivity analysis complete!")
    print(f"   Results saved to: {BASE_LOG_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()