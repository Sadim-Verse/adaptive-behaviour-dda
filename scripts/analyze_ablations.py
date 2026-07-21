# scripts/analyze_ablations.py
"""
Analyse ablation results and produce Table 5. DDQN only (matches the
table's own title -- ablations were never run for DQN/PPO).

FIXES APPLIED IN THIS VERSION:
1. Path construction for Full System now uses FLAT STRUCTURE for Single/Multi
   (no extra "ddqn" subfolder) — matching where run_all_evaluations.py
   actually saves the data.
2. Adaptation Latency now uses sample standard deviation (ddof=1) consistently
   with the established convention in run_delta_sensitivity.py.
3. FLOW_ZONES imported from training.train_multi_env (single source of truth).
4. Procedural (obstacle_minigrid) included with its own base log directory.
5. Sentinel values (>= 500) excluded from latency aggregation.
"""

import os
import time
import argparse
import numpy as np

try:
    from training.train_multi_env import FLOW_ZONES as _FLOW_ZONES_RAW
    FLOW_ZONES = {env: bounds["default"] for env, bounds in _FLOW_ZONES_RAW.items()}
except ImportError:
    print("WARNING: could not import FLOW_ZONES from training.train_multi_env "
          "(run as `python -m scripts.analyze_ablations` from the project root). "
          "Falling back to last-confirmed printed values.")
    FLOW_ZONES = {
        "single": (296.0, 765.0),
        "multi": (245.6, 470.1),
        "obstacle_minigrid": (-74.5, -23.7),
    }

ABLATIONS_BASE_DIR = "./logs/ablations"
FULL_SYSTEM_BASE_DIRS = {
    "single": "./logs/experiments",
    "multi": "./logs/experiments",
    "obstacle_minigrid": "./logs/procedural_env_v2",
}
ABLATIONS = ["no_dda", "fixed_difficulty", "no_rolling_window", "no_reward_smoothing"]
ENVS = ["single", "multi", "obstacle_minigrid"]
SEEDS = [42, 105, 777, 999, 2026]
WINDOW_SIZE = 10
N_EPISODES_SPIKE_RUN = 500


def ablation_folder(env, seed, ablation, spike=False):
    suffix = "no_smoothing" if ablation == "no_reward_smoothing" else ablation
    spike_part = "_spike" if spike else ""
    return f"{env}_ddqn_delta0p05_seed{seed}{spike_part}_abl_{suffix}"


def full_system_folder(env, seed, spike=False):
    suffix = "_spike" if spike else ""
    return f"{env}_ddqn_delta0p05_seed{seed}{suffix}"


def full_system_path(env, seed, spike=False):
    """
    FLAT STRUCTURE — no extra per-algo subfolder for any environment.
    This matches where run_all_evaluations.py saves the data.
    """
    base = FULL_SYSTEM_BASE_DIRS[env]
    folder = full_system_folder(env, seed, spike)
    return os.path.join(base, folder)


def compute_fzc(rolling_avgs, flow_low, flow_high, window_size=WINDOW_SIZE):
    if len(rolling_avgs) < window_size:
        return 0.0
    valid_avgs = rolling_avgs[window_size - 1:]
    in_zone = np.sum((valid_avgs >= flow_low) & (valid_avgs <= flow_high))
    return (in_zone / len(valid_avgs)) * 100.0


def compute_reward_fzc(folder_path, flow_low, flow_high):
    rewards_file = os.path.join(folder_path, "episode_rewards.npy")
    rolling_file = os.path.join(folder_path, "rolling_avg.npy")
    if not (os.path.exists(rewards_file) and os.path.exists(rolling_file)):
        return None
    rewards = np.load(rewards_file)
    rolling_avgs = np.load(rolling_file)
    return {
        "mean_reward": float(np.mean(rewards)),
        "fzc": compute_fzc(rolling_avgs, flow_low, flow_high),
        "mtime": os.path.getmtime(rewards_file),
    }


def censored_latency(folder_paths, n_episodes=N_EPISODES_SPIKE_RUN):
    """
    Sentinel-safe latency aggregation across a list of run folders.
    Uses sample standard deviation (ddof=1) consistently with
    run_delta_sensitivity.py.
    """
    latencies = []
    for path in folder_paths:
        f = os.path.join(path, "adaptation_latency.npy")
        if os.path.exists(f):
            v = np.load(f)
            if len(v) > 0:
                latencies.append(float(np.asarray(v).flatten()[0]))
    latencies = np.array(latencies)
    n_total = len(latencies)
    recovered = latencies[latencies < n_episodes] if n_total else latencies
    if n_total == 0:
        return None
    return {
        "mean": float(np.mean(recovered)) if len(recovered) else float("nan"),
        "std": float(np.std(recovered, ddof=1)) if len(recovered) > 1 else 0.0,
        "n_recovered": len(recovered),
        "n_total": n_total,
    }


def fmt_latency(lat):
    if lat is None:
        return "N/A"
    if lat["n_recovered"] == 0:
        return f"NEVER RECOVERED (0/{lat['n_total']})"
    s = f"{lat['mean']:.1f} +/- {lat['std']:.1f}"
    if lat["n_recovered"] < lat["n_total"]:
        s += f" ({lat['n_recovered']}/{lat['n_total']} recov.)"
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--envs", nargs="+", default=ENVS, choices=ENVS)
    args = ap.parse_args()

    print("=" * 80)
    print("TABLE 5 -- ABLATION STUDY RESULTS (DDQN)")
    print(f"Flow Zones (imported): {FLOW_ZONES}")
    print("=" * 80)

    print("\nChecking folder structure...")
    for ablation in ABLATIONS:
        p = os.path.join(ABLATIONS_BASE_DIR, ablation)
        print(f"  {'FOUND' if os.path.exists(p) else 'MISSING'}: {p}")

    all_rows = []

    for env in args.envs:
        print(f"\n{'-'*80}")
        print(f"ENVIRONMENT: {env.upper()}")
        print(f"{'-'*80}")

        flow_low, flow_high = FLOW_ZONES[env]

        # Full System -- read from real data, not hardcoded
        fs_steady_paths = [full_system_path(env, s, spike=False) for s in SEEDS]
        fs_spike_paths = [full_system_path(env, s, spike=True) for s in SEEDS]
        fs_metrics = [compute_reward_fzc(p, flow_low, flow_high) for p in fs_steady_paths]
        fs_metrics = [m for m in fs_metrics if m is not None]
        fs_latency = censored_latency(fs_spike_paths)

        if fs_metrics:
            fs_rewards = [m["mean_reward"] for m in fs_metrics]
            fs_fzcs = [m["fzc"] for m in fs_metrics]
            oldest_mtime = min(m["mtime"] for m in fs_metrics)
            print(f"{'Full System':<25} | {np.mean(fs_rewards):>8.1f} +/- {np.std(fs_rewards):<6.1f} | "
                  f"{np.mean(fs_fzcs):>6.1f}% +/- {np.std(fs_fzcs):<5.1f}% | latency: {fmt_latency(fs_latency)}"
                  f"  [oldest file: {time.ctime(oldest_mtime)}]")
            all_rows.append(("Full System", env, np.mean(fs_rewards), np.std(fs_rewards),
                              np.mean(fs_fzcs), np.std(fs_fzcs), fmt_latency(fs_latency)))
        else:
            print(f"{'Full System':<25} | NO DATA at {fs_steady_paths[0]} (check path)")

        for ablation in ABLATIONS:
            paths = [os.path.join(ABLATIONS_BASE_DIR, ablation, ablation_folder(env, s, ablation)) for s in SEEDS]
            metrics = [compute_reward_fzc(p, flow_low, flow_high) for p in paths]
            found_paths = [p for p, m in zip(paths, metrics) if m is not None]
            metrics = [m for m in metrics if m is not None]

            label = ablation.replace("_", " ").title()
            if not metrics:
                print(f"{label:<25} | NO DATA (checked {len(paths)} seed folders)")
                continue

            rewards = [m["mean_reward"] for m in metrics]
            fzcs = [m["fzc"] for m in metrics]
            oldest_mtime = min(m["mtime"] for m in metrics)

            # Latency only meaningful for conditions with spike-mode data;
            # no_dda / fixed_difficulty / etc. may not have a spike variant
            # depending on how the ablation sweep was run -- check, don't assume.
            spike_paths = [os.path.join(ABLATIONS_BASE_DIR, ablation,
               ablation_folder(env, s, ablation, spike=True)) for s in SEEDS]

            lat = censored_latency(spike_paths) 

            print(f"{label:<25} | {np.mean(rewards):>8.1f} +/- {np.std(rewards):<6.1f} | "
                  f"{np.mean(fzcs):>6.1f}% +/- {np.std(fzcs):<5.1f}% | latency: {fmt_latency(lat)}"
                  f"  [oldest file: {time.ctime(oldest_mtime)}]")

            all_rows.append((label, env, np.mean(rewards), np.std(rewards),
                              np.mean(fzcs), np.std(fzcs), fmt_latency(lat)))

    print("\n" + "=" * 80)
    print("REMINDER: cross-check the '[oldest file: ...]' timestamps above against")
    print("when envs/chase_env.py and envs/multi_chase_env.py's clip fix (6.0 -> 9.0)")
    print("was applied. Any condition with data older than that fix -- especially")
    print("fixed_difficulty -- needs to be re-run, not just re-aggregated.")
    print("=" * 80)


if __name__ == "__main__":
    main()