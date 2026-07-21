# scripts/compile_results.py
"""
Table 3 / Table 8 generation: aggregates the main evaluation battery
(steady + spike mode) across DQN/DDQN/PPO x Single/Multi/Procedural x
5 seeds into per-algorithm, per-environment summary statistics.

This is a rewrite of an earlier version, which was found to have several
real bugs, not just missing statistics -- fixed here:

1. STALE FLOW ZONE BOUNDS. The old version hardcoded
   get_flow_bounds() = (600,1200) for single, (220,470) for multi -- the
   pre-Priority-3 placeholder values, not the calibrated bounds used
   everywhere else in this project. Fixed: FLOW_ZONES is now imported
   DIRECTLY from training.train_multi_env (single source of truth), so
   every FZC number this script produces uses the real, current bounds
   (Single: 296.0-765.0, Multi: 245.6-470.1, Procedural: -74.5 to -23.7),
   and can never drift out of sync with the rest of the project again.

2. WRONG FOLDER PATH. The old version looked for evaluation data at
   BASE_LOG_DIR/{algo}/{env}_{algo}_delta.../ -- an extra algo-named
   subfolder. Confirmed with the project owner: evaluation output
   (episode_rewards.npy etc.) is written FLAT, e.g.
   ./logs/experiments/single_ddqn_delta0p05_seed42_spike/, with NO algo
   subfolder. (The algo-subfolder pattern is real, but it's how TRAINING
   output -- best_model.zip -- is organized, not evaluation output.) The
   old path pattern meant every os.path.exists() check silently failed --
   no error, no data, and the printed table would have been built from
   whatever partial/empty lists resulted. Fixed: run_folder_path() now
   matches the actual eval-output layout.

3. PROCEDURAL WAS MISSING. Added obstacle_minigrid as a third
   environment, matching every other table/script in this project
   (run_all_evaluations.py, run_delta_sensitivity.py, calibrate_flow_
   zones.py). It uses a different base log directory
   (./logs/procedural_env_v2) -- the same real folder-layout
   inconsistency already documented and handled elsewhere.

4. LATENCY SENTINEL CONTAMINATION (Priority 6). A seed that never
   recovers within the 500-episode run reports adaptation_latency=500 as
   a SENTINEL value, not a genuine duration -- mathematically, a genuine
   recovery latency can never reach 500 (t_drop=250, run caps at 500
   episodes -> max possible real latency is 250). The old version
   averaged this straight into np.mean()/np.std(), badly distorting both.
   Fixed: reuses the identical censored_latency_stats() function already
   verified in run_delta_sensitivity.py -- reports recovery_rate
   separately and computes latency mean/std/CI only over seeds that
   actually recovered.

5. CONFIDENCE INTERVALS (Priority 4, the actual ask that started this
   rewrite). Bootstrap 95% CIs (percentile method, 10,000 resamples)
   added for mean_reward, fzc, and (censored) adaptation latency.
"""

import os
import numpy as np

from training.train_multi_env import FLOW_ZONES as _FLOW_ZONES_RAW

ENVS = ["single", "multi", "obstacle_minigrid"]
ALGOS = ["dqn", "ddqn", "ppo"]
SEEDS = [42, 105, 777, 999, 2026]
MODES = ["steady", "spike"]
WINDOW_SIZE = 10
N_EPISODES_SPIKE_RUN = 500  # run_dda_only()'s total episode budget --
                             # the censoring cutoff for adaptation_latency,
                             # see fix #4 above.
N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95

# Fix #1: single source of truth, not a local hardcoded copy.
FLOW_ZONES = {env: bounds["default"] for env, bounds in _FLOW_ZONES_RAW.items()}

# Fix #2/#3: base directory differs for obstacle_minigrid; eval output is
# FLAT (no algo subfolder) for every environment.
BASE_LOG_DIRS = {
    "single": "./logs/experiments",
    "multi": "./logs/experiments",
    "obstacle_minigrid": "./logs/procedural_env_v2",
}
ENV_LABELS = {"single": "Single", "multi": "Multi", "obstacle_minigrid": "Procedural"}


def run_folder_path(env, algo, seed, mode):
    suffix = "_spike" if mode == "spike" else ""
    run_folder = f"{env}_{algo}_delta0p05_seed{seed}{suffix}"
    return os.path.join(BASE_LOG_DIRS[env], run_folder)


def calculate_fzc(rolling_avgs, flow_low, flow_high):
    if len(rolling_avgs) < WINDOW_SIZE:
        return 0.0
    valid_avgs = rolling_avgs[WINDOW_SIZE - 1:]
    in_zone = np.sum((valid_avgs >= flow_low) & (valid_avgs <= flow_high))
    return (in_zone / len(valid_avgs)) * 100.0


def load_adaptation_latency(folder_path):
    latency_file = os.path.join(folder_path, "adaptation_latency.npy")
    if os.path.exists(latency_file):
        latency = np.load(latency_file)
        if len(latency) > 0:
            return float(np.asarray(latency).flatten()[0])
    return None


def bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP, ci_level=CI_LEVEL, seed=0):
    """Percentile-method bootstrap CI -- preferred over a normal-assumption
    t-interval since n=5 seeds doesn't justify a normality assumption."""
    values = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = 1 - ci_level
    lo = float(np.percentile(boot_means, 100 * (alpha / 2)))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def censored_latency_stats(latencies, n_episodes=N_EPISODES_SPIKE_RUN,
                            n_bootstrap=N_BOOTSTRAP, ci_level=CI_LEVEL, seed=0):
    """Fix #4 -- identical logic to run_delta_sensitivity.py's version
    (already tested there against 6 cases including the real Multi
    delta=0.3 example and boundary edge cases), reused here so the same
    bug gets the same fix everywhere it appears in this project."""
    latencies = np.asarray([v for v in latencies if not np.isnan(v)], dtype=float)
    n_seeds = len(latencies)
    out = dict(n_seeds=n_seeds, n_recovered=0, recovery_rate=float("nan"),
               latency_mean=float("nan"), latency_std=float("nan"),
               latency_ci95_lo=float("nan"), latency_ci95_hi=float("nan"))
    if n_seeds == 0:
        return out
    recovered = latencies[latencies < n_episodes]
    n_recovered = len(recovered)
    out["n_recovered"] = n_recovered
    out["recovery_rate"] = n_recovered / n_seeds
    if n_recovered == 0:
        return out
    out["latency_mean"] = float(np.mean(recovered))
    out["latency_std"] = float(np.std(recovered, ddof=1)) if n_recovered > 1 else 0.0
    if n_recovered > 1:
        out["latency_ci95_lo"], out["latency_ci95_hi"] = bootstrap_ci(
            recovered, n_bootstrap=n_bootstrap, ci_level=ci_level, seed=seed)
    else:
        out["latency_ci95_lo"] = out["latency_ci95_hi"] = out["latency_mean"]
    return out


def main():
    print("=" * 100)
    print("AGGREGATING MULTI-SEED DDA BENCHMARK DATA  (Table 3 / Table 8 source)")
    print("=" * 100)

    results_by_mode = {}

    for mode in MODES:
        print(f"\n{'='*100}")
        print(f"MODE: {mode.upper()}")
        print(f"{'='*100}")

        results_by_mode[mode] = {}

        for env in ENVS:
            print(f"\nENVIRONMENT PROFILE: {ENV_LABELS[env].upper()}")
            print("-" * 100)
            if mode == "spike":
                print(f"{'Algorithm':<10} | {'Mean Reward [95% CI]':<26} | {'FZC (%) [95% CI]':<24} | "
                      f"{'Latency (recovered only)':<26} | {'Recovery':<10}")
            else:
                print(f"{'Algorithm':<10} | {'Mean Reward [95% CI]':<26} | {'FZC (%) [95% CI]':<24}")
            print("-" * 100)

            flow_low, flow_high = FLOW_ZONES[env]
            results_by_mode[mode][env] = {}

            for algo in ALGOS:
                algo_rewards, algo_fzcs, algo_latencies = [], [], []
                n_found = 0

                for seed in SEEDS:
                    folder_path = run_folder_path(env, algo, seed, mode)
                    rewards_file = os.path.join(folder_path, "episode_rewards.npy")
                    rolling_file = os.path.join(folder_path, "rolling_avg.npy")

                    if os.path.exists(rewards_file) and os.path.exists(rolling_file):
                        n_found += 1
                        rewards = np.load(rewards_file)
                        rolling_avgs = np.load(rolling_file)
                        algo_rewards.append(np.mean(rewards))
                        algo_fzcs.append(calculate_fzc(rolling_avgs, flow_low, flow_high))
                        if mode == "spike":
                            latency = load_adaptation_latency(folder_path)
                            algo_latencies.append(latency if latency is not None else float("nan"))

                if n_found < len(SEEDS):
                    example_path = run_folder_path(env, algo, SEEDS[0], mode)
                    print(f"  NOTE: {algo.upper()} found only {n_found}/{len(SEEDS)} seed runs "
                          f"(checked e.g. {example_path})")

                if not algo_rewards:
                    print(f"{algo.upper():<10} | NO DATA FOUND")
                    continue

                rew_mean, rew_std = np.mean(algo_rewards), np.std(algo_rewards)
                rew_ci = bootstrap_ci(algo_rewards)
                fzc_mean, fzc_std = np.mean(algo_fzcs), np.std(algo_fzcs)
                fzc_ci = bootstrap_ci(algo_fzcs)

                entry = {
                    "n_seeds_found": n_found,
                    "rew_mean": rew_mean, "rew_std": rew_std,
                    "rew_ci95_lo": rew_ci[0], "rew_ci95_hi": rew_ci[1],
                    "fzc_mean": fzc_mean, "fzc_std": fzc_std,
                    "fzc_ci95_lo": fzc_ci[0], "fzc_ci95_hi": fzc_ci[1],
                }

                rew_str = f"{rew_mean:>6.1f} [{rew_ci[0]:.1f},{rew_ci[1]:.1f}]"
                fzc_str = f"{fzc_mean:>5.1f}% [{fzc_ci[0]:.1f},{fzc_ci[1]:.1f}]"

                if mode == "spike":
                    lat_stats = censored_latency_stats(algo_latencies)
                    entry.update({
                        "lat_mean": lat_stats["latency_mean"],
                        "lat_std": lat_stats["latency_std"],
                        "lat_ci95_lo": lat_stats["latency_ci95_lo"],
                        "lat_ci95_hi": lat_stats["latency_ci95_hi"],
                        "recovery_rate": lat_stats["recovery_rate"],
                        "n_recovered": lat_stats["n_recovered"],
                    })
                    if np.isnan(entry["lat_mean"]):
                        lat_str = "N/A (0 recovered)"
                    else:
                        lat_str = f"{entry['lat_mean']:.1f}+/-{entry['lat_std']:.1f}"
                    rec_str = f"{entry['n_recovered']}/{n_found}"
                    print(f"{algo.upper():<10} | {rew_str:<26} | {fzc_str:<24} | {lat_str:<26} | {rec_str:<10}")
                else:
                    print(f"{algo.upper():<10} | {rew_str:<26} | {fzc_str:<24}")

                results_by_mode[mode][env][algo] = entry

    print("\n" + "=" * 100)
    print("SUMMARY COMPARISON: STEADY vs SPIKE")
    print("=" * 100)

    for env in ENVS:
        print(f"\nENVIRONMENT: {ENV_LABELS[env].upper()}")
        print("-" * 115)
        print(f"{'Algorithm':<10} | {'Mode':<8} | {'Mean Reward [95% CI]':<26} | "
              f"{'FZC (%) [95% CI]':<24} | {'Latency (eps)':<20}")
        print("-" * 115)

        for algo in ALGOS:
            for mode in MODES:
                if mode in results_by_mode and env in results_by_mode[mode] and algo in results_by_mode[mode][env]:
                    data = results_by_mode[mode][env][algo]
                    mode_label = "Steady" if mode == "steady" else "Spike"
                    rew_str = f"{data['rew_mean']:>6.1f} [{data['rew_ci95_lo']:.1f},{data['rew_ci95_hi']:.1f}]"
                    fzc_str = f"{data['fzc_mean']:>5.1f}% [{data['fzc_ci95_lo']:.1f},{data['fzc_ci95_hi']:.1f}]"
                    if mode == "spike":
                        lat_str = "N/A" if np.isnan(data["lat_mean"]) else f"{data['lat_mean']:.1f}+/-{data['lat_std']:.1f}"
                    else:
                        lat_str = "N/A (no spike)"
                    print(f"{algo.upper():<10} | {mode_label:<8} | {rew_str:<26} | {fzc_str:<24} | {lat_str:<20}")

    print("\n" + "=" * 100)
    print("LEGEND:")
    print("  Flow Zone bounds (imported from train_multi_env.py, single source of truth):")
    for env in ENVS:
        print(f"    {ENV_LABELS[env]}: {FLOW_ZONES[env]}")
    print("  Adaptation Latency = mean/std computed ONLY over seeds that recovered")
    print("    within the 500-episode window (Priority 6 fix). See the Recovery")
    print("    column for how many of the found seeds actually recovered -- a")
    print("    partial recovery rate (e.g. 4/5) should be reported explicitly,")
    print("    not hidden inside the mean.")
    print("  Mean Reward / FZC / recovered-only Latency all include a bootstrapped")
    print("    95% CI (percentile method, 10,000 resamples) -- Priority 4.")
    print("=" * 100)


if __name__ == "__main__":
    main()