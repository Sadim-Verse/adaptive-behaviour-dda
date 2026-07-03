# scripts/compile_results.py
import os
import numpy as np

ENVS = ["single", "multi"]
ALGOS = ["dqn", "ddqn", "ppo"]
SEEDS = [42, 105, 777, 999, 2026]
MODES = ["steady", "spike"]

# ── NEW: Algorithm folder mapping ────────────────────────────────────────── #
ALGO_FOLDERS = {
    "dqn": "dqn",
    "ddqn": "ddqn",
    "ppo": "ppo"
}

BASE_LOG_DIR = "./logs/experiments"

def get_flow_bounds(env):
    if env == "multi":
        return 220.0, 470.0
    return 600.0, 1200.0

WINDOW_SIZE = 10

def calculate_fzc(rolling_avgs, flow_low, flow_high):
    if len(rolling_avgs) < WINDOW_SIZE:
        return 0.0
    valid_avgs = rolling_avgs[WINDOW_SIZE - 1:]
    in_zone = np.sum((valid_avgs >= flow_low) & (valid_avgs <= flow_high))
    return (in_zone / len(valid_avgs)) * 100.0

def load_adaptation_latency(folder_path):
    """Load the adaptation latency from the saved .npy file."""
    latency_file = os.path.join(folder_path, "adaptation_latency.npy")
    if os.path.exists(latency_file):
        latency = np.load(latency_file)
        if len(latency) > 0:
            return float(latency[0])
    return None

print("=" * 85)
print("AGGREGATING MULTI-SEED DDA BENCHMARK DATA")
print("=" * 85)

results_by_mode = {}

for mode in MODES:
    print(f"\n{'='*85}")
    print(f"📊 MODE: {mode.upper()}")
    print(f"{'='*85}")

    results_by_mode[mode] = {}

    for env in ENVS:
        print(f"\n📊 ENVIRONMENT PROFILE: {env.upper()}")
        print("-" * 85)

        if mode == "spike":
            print(f"{'Algorithm':<12} | {'Mean Reward':<18} | {'FZC (%)':<18} | {'Adaptation Latency':<18}")
        else:
            print(f"{'Algorithm':<12} | {'Mean Reward':<18} | {'FZC (%)':<18}")
        print("-" * 85)

        FLOW_LOW, FLOW_HIGH = get_flow_bounds(env)
        results_by_mode[mode][env] = {}

        for algo in ALGOS:
            algo_rewards = []
            algo_fzcs = []
            algo_latencies = []

            for seed in SEEDS:
                # ── NEW: Build path with algorithm subfolder ─────────────── #
                algo_folder = ALGO_FOLDERS[algo]
                if mode == "spike":
                    run_folder = f"{env}_{algo}_delta0p05_seed{seed}_spike"
                else:
                    run_folder = f"{env}_{algo}_delta0p05_seed{seed}"

                folder_path = os.path.join(BASE_LOG_DIR, algo_folder, run_folder)
                rewards_file = os.path.join(folder_path, "episode_rewards.npy")
                rolling_file = os.path.join(folder_path, "rolling_avg.npy")

                if os.path.exists(rewards_file) and os.path.exists(rolling_file):
                    rewards = np.load(rewards_file)
                    rolling_avgs = np.load(rolling_file)

                    mean_rew = np.mean(rewards)
                    fzc = calculate_fzc(rolling_avgs, FLOW_LOW, FLOW_HIGH)

                    algo_rewards.append(mean_rew)
                    algo_fzcs.append(fzc)

                    if mode == "spike":
                        latency = load_adaptation_latency(folder_path)
                        if latency is None:
                            latency = float('nan')
                        algo_latencies.append(latency)

            if algo_rewards:
                rew_mean, rew_std = np.mean(algo_rewards), np.std(algo_rewards)
                fzc_mean, fzc_std = np.mean(algo_fzcs), np.std(algo_fzcs)

                results_by_mode[mode][env][algo] = {
                    "rew_mean": rew_mean,
                    "rew_std": rew_std,
                    "fzc_mean": fzc_mean,
                    "fzc_std": fzc_std,
                }

                if mode == "spike":
                    lat_mean, lat_std = np.mean(algo_latencies), np.std(algo_latencies)
                    results_by_mode[mode][env][algo]["lat_mean"] = lat_mean
                    results_by_mode[mode][env][algo]["lat_std"] = lat_std
                    print(f"{algo.upper():<12} | "
                          f"{rew_mean:>6.1f} ± {rew_std:<5.1f} | "
                          f"{fzc_mean:>5.1f}% ± {fzc_std:<4.1f}% | "
                          f"{lat_mean:>5.1f} ± {lat_std:<4.1f} eps")
                else:
                    print(f"{algo.upper():<12} | "
                          f"{rew_mean:>6.1f} ± {rew_std:<5.1f} | "
                          f"{fzc_mean:>5.1f}% ± {fzc_std:<4.1f}%")

print("\n" + "=" * 85)
print("SUMMARY COMPARISON: STEADY vs SPIKE")
print("=" * 85)

for env in ENVS:
    print(f"\n📊 ENVIRONMENT: {env.upper()}")
    print("-" * 95)
    print(f"{'Algorithm':<12} | {'Mode':<8} | {'Mean Reward':<15} | {'FZC (%)':<15} | {'Latency (eps)':<12}")
    print("-" * 95)

    for algo in ALGOS:
        for mode in MODES:
            if mode in results_by_mode and env in results_by_mode[mode] and algo in results_by_mode[mode][env]:
                data = results_by_mode[mode][env][algo]
                mode_label = "Steady" if mode == "steady" else "Spike"
                if mode == "spike":
                    lat_str = f"{data['lat_mean']:>5.1f} ± {data['lat_std']:<4.1f}"
                else:
                    lat_str = "N/A (no spike)"
                print(f"{algo.upper():<12} | {mode_label:<8} | "
                      f"{data['rew_mean']:>6.1f} ± {data['rew_std']:<4.1f} | "
                      f"{data['fzc_mean']:>5.1f}% ± {data['fzc_std']:<4.1f}% | "
                      f"{lat_str}")

print("\n" + "=" * 85)
print("LEGEND:")
print("  Steady Mode : max_speed=9.0, NO spike protocol")
print("  Spike Mode  : max_speed=9.0, WITH spike at episode 250")
print("  Adaptation Latency = Formal metric (recovery from spike) — ONLY for SPIKE mode")
print("  Steady mode has N/A for Latency (no spike → no recovery to measure)")
print("=" * 85)