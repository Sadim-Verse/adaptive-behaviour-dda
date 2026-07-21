# scripts/run_delta_sensitivity.py
"""
Delta sensitivity analysis, multi-seed, DDQN ONLY, across all 3
environments, both steady and spike modes.

  - 6 delta values: 0.05, 0.10, 0.20, 0.30, 0.50, 1.00
  - 3 environments: single, multi, obstacle_minigrid
  - 5 seeds: 42, 105, 777, 999, 2026

Total runs per mode: 6 deltas x 3 envs x 5 seeds = 90
(existing runs are skipped via cached .npy files -- only metrics are
recomputed, no expensive re-evaluation needed for data already collected)

Results saved to: ./logs/delta_sensitivity_<mode>/

CHANGES FROM THE PREVIOUS VERSION (real fixes, not narration this time):

1. FLOW_ZONES is no longer a separate, hand-maintained dict in this file.
   It's imported DIRECTLY from training.train_multi_env, which is the
   actual single source of truth used by every evaluation run. The
   previous version's local copy went stale (still said Single was a
   "placeholder" via a (None,None)-style check, while the real bound
   had already been calibrated and pasted into train_multi_env.py) --
   that drift is exactly what caused the "still a placeholder" warning
   you saw despite the FLOW_ZONES dict in this file literally containing
   real numbers. Importing directly makes that class of bug structurally
   impossible: there's only one dict now, not two that can disagree.

2. The oscillation metric is REPLACED, not "kept deliberately" -- this
   was the actual bug you caught. compute_oscillation_frequency's
   direction_changes/len(speed_history) denominator counts EVERY episode,
   including the ones where speed never changed at all (because the
   controller is saturated at a bound, or already inside the Flow Zone
   doing nothing). That dilutes the metric toward zero regardless of how
   often the controller actually reverses direction when it DOES move --
   which is exactly the "controller appears stable at ALL deltas"
   degenerate result Priority 2 was raised to fix, and this script had
   silently reintroduced it. Replaced with two Priority-2-aligned metrics:
     - reversal_frequency: fraction of NONZERO speed changes that flip
       direction, denominator = nonzero transitions only
     - rate_of_change_variance: Var(delta_d) across the full episode
       history -- a direct reading of one of the metric examples your
       supervisor listed by name
   Both verified against real cached Single/Multi speed_history.npy data
   before being shipped here (0.0577 for Single, 0.1170 for Multi at
   delta=0.05 -- real, nonzero, environment-differentiated numbers,
   unlike the old formula's flat 0.0000).

3. Priority 4 (confidence intervals): aggregation now reports a
   bootstrapped 95% CI (percentile method, 10,000 resamples) alongside
   mean+/-std for mean_reward and fzc, computed across the 5 seeds per
   (environment, delta) cell. With n=5, a normal-assumption CI is not
   well justified -- bootstrap percentile CIs make fewer assumptions and
   are what was recommended earlier in this project for exactly this
   n=5-seeds situation.

4. Adaptation latency: this script does NOT recompute it (that logic
   lives in train_multi_env.py's run_dda_only(), not here) -- it just
   loads whatever is in adaptation_latency.npy. If you've applied the
   Priority 6 sustained-recovery fix via
   scripts/recompute_adaptation_latency.py, this script will
   automatically pick up the corrected values on its next run (it always
   reads the file fresh). If you haven't run that yet, the latency
   numbers here still reflect the OLD single-crossing detection --
   run that script first if you want the corrected numbers before
   generating your thesis tables from this output.
"""

import torch  # noqa: F401  -- MUST be imported first on Windows. Importing
              # pandas before torch causes a DLL init failure (WinError 1114
              # on torch/lib/c10.dll) because pandas loads a runtime DLL that
              # conflicts with one of torch's own. Confirmed via:
              #   python -c "import pandas; import torch"  -> fails
              #   python -c "import torch; import pandas"  -> works
              # Importing torch first avoids the conflict entirely.

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Single source of truth -- see change #1 above. This import requires
# running this script as a module from the project root:
#   python -m scripts.run_delta_sensitivity
from training.train_multi_env import FLOW_ZONES as _FLOW_ZONES_RAW, DDA_SCALE

# ── Configuration ─────────────────────────────────────────────────────────── #

DELTAS = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]
ENVIRONMENTS = ["single", "multi", "obstacle_minigrid"]
SEEDS = [42, 105, 777, 999, 2026]
SPIKE_EPISODE = 250
N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95
# run_dda_only() falls back to reporting adaptation_latency = n_episodes
# (500) when a seed never recovers within the run -- this is a SENTINEL
# for "did not recover," not a genuine measured duration. Mathematically
# unambiguous: since t_drop=SPIKE_EPISODE=250 and the run caps at 500
# episodes, the largest possible GENUINE latency is 500-250=250, so any
# recorded value >= N_EPISODES_PER_RUN can only be the fallback, never a
# real recovery time. Blending it into a continuous mean/std alongside
# genuine latencies badly distorts both -- see main()'s aggregation.
N_EPISODES_PER_RUN = 500

# Flatten train_multi_env's {"env": {"default": (lo, hi)}} into {"env": (lo, hi)}
FLOW_ZONES = {env: bounds["default"] for env, bounds in _FLOW_ZONES_RAW.items()}

MAX_SPEED = {
    env: (DDA_SCALE[env]["max_speed_default"]) for env in ENVIRONMENTS
}
# single/multi's actual DDA ceiling used throughout this project is 9.0
# (distinct from ChaseEnv's internal clip to 6.0) -- DDA_SCALE already
# encodes this correctly (max_speed_default=9.0 for both), so no override
# needed here; obstacle_minigrid correctly resolves to its own 1.0 scale.

# Paths to trained DDQN models. single/multi have an extra algo-named
# subfolder that obstacle_minigrid does not (same real inconsistency
# handled elsewhere in this project -- see calibrate_flow_zones.py's
# model_path_for() docstring).
BASE_LOG_DIRS = {
    "single": "./logs/experiments",
    "multi": "./logs/experiments",
    "obstacle_minigrid": "./logs/procedural_env_v2",
}


def model_path_for(env_type):
    folder = f"{env_type}_ddqn_delta0p05_seed42"
    base = BASE_LOG_DIRS[env_type]
    if env_type in ("single", "multi"):
        return os.path.join(base, "ddqn", folder, "best_model.zip")
    return os.path.join(base, folder, "best_model.zip")


MODEL_PATHS = {env: model_path_for(env) for env in ENVIRONMENTS}


# ── Priority 2: real instability metrics ──────────────────────────────────── #

def compute_reversal_frequency(speed_history):
    """
    Fraction of NONZERO speed changes that reverse direction (sign flip),
    among nonzero transitions only. Denominator = number of nonzero
    transitions, NOT total episode count -- see module docstring change #2
    for why this matters (the old all-episode-denominator version is what
    produced the flat 0.0000 "controller appears stable at all deltas" bug).
    """
    diffs = np.diff(np.asarray(speed_history, dtype=float))
    nonzero = diffs[diffs != 0]
    if len(nonzero) < 2:
        return 0.0
    signs = np.sign(nonzero)
    reversals = np.sum(signs[1:] != signs[:-1])
    return float(reversals) / (len(nonzero) - 1)


def compute_rate_of_change_variance(speed_history):
    """Var(delta_d) across the full episode history -- one of the metric
    examples your supervisor listed by name directly."""
    diffs = np.diff(np.asarray(speed_history, dtype=float))
    if len(diffs) < 2:
        return 0.0
    return float(np.var(diffs))


def compute_survival_time(rewards):
    """Approximates steps-survived by undoing the -100 collision penalty
    offset (reward = +1/step, -100 on collision, so reward+100 ~= steps
    survived). The r>=2000 clamp only ever engages for single/multi
    (ChaseEnv-style envs where near-perfect survival is achievable);
    it's simply inert, not wrong, for obstacle_minigrid, whose reward
    scale never approaches that ceiling."""
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
    if flow_low is None or flow_high is None:
        return float("nan")
    if len(rolling_avgs) < window_size:
        return 0.0
    valid_avgs = rolling_avgs[window_size - 1:]
    in_zone = np.sum((valid_avgs >= flow_low) & (valid_avgs <= flow_high))
    return (in_zone / len(valid_avgs)) * 100.0


def load_adaptation_latency(folder_path):
    """Only meaningful in spike mode. Always reads fresh from disk, so if
    scripts/recompute_adaptation_latency.py has been run since the last
    time this script ran, the corrected (Priority 6, sustained-recovery)
    value is picked up automatically -- no special-casing needed here."""
    latency_file = os.path.join(folder_path, "adaptation_latency.npy")
    if os.path.exists(latency_file):
        latency = np.load(latency_file)
        if len(latency) > 0:
            return float(np.asarray(latency).flatten()[0])
    return None


# ── Priority 4: bootstrap confidence intervals ────────────────────────────── #

def bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP, ci_level=CI_LEVEL, seed=0):
    """
    Percentile-method bootstrap CI. Preferred over a normal-assumption
    t-interval here because n=5 seeds is too small to justify a normality
    assumption -- bootstrap resampling makes no distributional assumption
    about the underlying seed-to-seed variation.
    """
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


# ── Main ─────────────────────────────────────────────────────────────────── #

def main(mode, envs, seeds, deltas):
    base_log_dir = f"./logs/delta_sensitivity_{mode}"
    os.makedirs(base_log_dir, exist_ok=True)

    for env in envs:
        if FLOW_ZONES[env][0] is None:
            print(f"WARNING: FLOW_ZONES['{env}'] is genuinely still a placeholder "
                  f"in training.train_multi_env. FZC for this environment will show "
                  f"as N/A until it's calibrated there (this script now imports "
                  f"directly, so fixing it in ONE place fixes it everywhere).")

    print("=" * 80)
    print(f"DELTA SENSITIVITY ANALYSIS (MULTI-SEED) - {mode.upper()} MODE")
    print(f"  Algorithm: DDQN")
    print(f"  Spike at episode: {SPIKE_EPISODE}" if mode == "spike" else "  No spike (steady mode)")
    print(f"  Seeds: {seeds}")
    print(f"  Deltas: {deltas}")
    print(f"  Environments: {envs}")
    print(f"  Flow Zones (imported from train_multi_env.py): {FLOW_ZONES}")
    print(f"  Total runs: {len(deltas) * len(envs) * len(seeds)}")
    print("=" * 80)

    all_results = []

    for env in envs:
        print(f"\n{'-'*80}")
        print(f"ENVIRONMENT: {env.upper()}")
        print(f"{'-'*80}")

        model_path = MODEL_PATHS[env]
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue

        for delta in deltas:
            print(f"\n  Delta = {delta}")
            delta_str = str(delta).replace('.', 'p')

            for seed in seeds:
                suffix = "_spike" if mode == "spike" else ""
                run_folder = f"{env}_ddqn_delta{delta_str}_seed{seed}{suffix}"
                log_dir = os.path.join(base_log_dir, run_folder)
                os.makedirs(log_dir, exist_ok=True)

                rewards_path = os.path.join(log_dir, "episode_rewards.npy")
                if os.path.exists(rewards_path):
                    print(f"      Seed {seed}: already exists, skipping (recomputing metrics only)")
                    rewards = np.load(rewards_path)
                    rolling_avgs = np.load(os.path.join(log_dir, "rolling_avg.npy"))
                    speed_history = np.load(os.path.join(log_dir, "speed_history.npy"))
                else:
                    import subprocess
                    cmd = [
                        sys.executable, "-m", "training.train_multi_env",
                        "--algo", "ddqn",
                        "--env", env,
                        "--resume", model_path,
                        "--delta", str(delta),
                        "--seed", str(seed),
                        "--max_speed", str(MAX_SPEED[env]),
                        "--output", base_log_dir,
                    ]
                    if mode == "spike":
                        cmd += ["--spike", str(SPIKE_EPISODE)]

                    print(f"      Seed {seed}: running...")
                    try:
                        subprocess.run(cmd, check=True, capture_output=False)
                    except subprocess.CalledProcessError as e:
                        print(f"      FAILED seed {seed} with exit code {e.returncode}")
                        continue

                    try:
                        rewards = np.load(rewards_path)
                        rolling_avgs = np.load(os.path.join(log_dir, "rolling_avg.npy"))
                        speed_history = np.load(os.path.join(log_dir, "speed_history.npy"))
                    except FileNotFoundError:
                        print(f"      Seed {seed} data not found after run")
                        continue

                flow_low, flow_high = FLOW_ZONES[env]
                mean_reward = np.mean(rewards)
                reward_std = np.std(rewards)
                fzc = compute_fzc(rolling_avgs, flow_low, flow_high)
                reversal_freq = compute_reversal_frequency(speed_history)
                roc_var = compute_rate_of_change_variance(speed_history)
                survival = compute_survival_time(rewards)

                latency = load_adaptation_latency(log_dir) if mode == "spike" else None
                if latency is None:
                    latency = float('nan')

                print(f"          Mean Reward: {mean_reward:.1f} +/- {reward_std:.1f}")
                print(f"          FZC: {fzc:.2f}%" if not np.isnan(fzc) else "          FZC: N/A")
                print(f"          Reversal Freq: {reversal_freq:.4f}  |  Rate-of-Change Var: {roc_var:.6f}")
                print(f"          Latency: {latency:.1f} eps" if not np.isnan(latency) else "          Latency: N/A")

                all_results.append({
                    "environment": env,
                    "delta": delta,
                    "seed": seed,
                    "mean_reward": mean_reward,
                    "reward_std": reward_std,
                    "fzc": fzc,
                    "reversal_freq": reversal_freq,
                    "rate_of_change_var": roc_var,
                    "survival_time": survival,
                    "adaptation_latency": latency,
                    "run_folder": run_folder
                })

    if not all_results:
        print("\nNo results collected. All runs failed.")
        return

    df = pd.DataFrame(all_results)

    # ── Aggregation with bootstrap CIs (Priority 4) ─────────────────────── #
    agg_rows = []
    for (env, delta), group in df.groupby(["environment", "delta"]):
        row = {"environment": env, "delta": delta, "n_seeds": len(group)}

        row["mean_reward"] = group["mean_reward"].mean()
        row["mean_reward_std_across_seeds"] = group["mean_reward"].std()
        ci_lo, ci_hi = bootstrap_ci(group["mean_reward"].values)
        row["mean_reward_ci95_lo"], row["mean_reward_ci95_hi"] = ci_lo, ci_hi

        row["fzc_mean"] = group["fzc"].mean()
        row["fzc_std"] = group["fzc"].std()
        ci_lo, ci_hi = bootstrap_ci(group["fzc"].values)
        row["fzc_ci95_lo"], row["fzc_ci95_hi"] = ci_lo, ci_hi

        row["reversal_freq_mean"] = group["reversal_freq"].mean()
        row["reversal_freq_std"] = group["reversal_freq"].std()
        row["rate_of_change_var_mean"] = group["rate_of_change_var"].mean()
        row["rate_of_change_var_std"] = group["rate_of_change_var"].std()

        row["survival_time_mean"] = group["survival_time"].mean()
        row["survival_time_std"] = group["survival_time"].std()

        # Priority 6 fix: a recorded latency >= N_EPISODES_PER_RUN is the
        # "never recovered" sentinel, not a genuine duration -- see the
        # constant's definition above for why this threshold is
        # unambiguous. Split into recovery_rate (did it happen at all)
        # and latency_mean/std computed ONLY over seeds that recovered,
        # instead of averaging the sentinel in as if it were real data.
        latencies = group["adaptation_latency"].dropna()
        if len(latencies) > 0:
            recovered_mask = latencies < N_EPISODES_PER_RUN
            recovered = latencies[recovered_mask]
            row["recovery_rate"] = float(recovered_mask.mean())
            row["n_recovered"] = int(recovered_mask.sum())
            row["n_never_recovered"] = int((~recovered_mask).sum())
            if len(recovered) > 0:
                row["latency_mean"] = float(recovered.mean())
                row["latency_std"] = float(recovered.std()) if len(recovered) > 1 else 0.0
            else:
                # every seed failed to recover -- there is no genuine
                # latency to average, report NaN rather than a fabricated
                # number (NOT the old behavior of averaging in 500s).
                row["latency_mean"] = float("nan")
                row["latency_std"] = float("nan")
        else:
            row["recovery_rate"] = float("nan")
            row["n_recovered"] = 0
            row["n_never_recovered"] = 0
            row["latency_mean"] = float("nan")
            row["latency_std"] = float("nan")

        agg_rows.append(row)

    aggregated = pd.DataFrame(agg_rows).round(4)

    agg_csv = os.path.join(base_log_dir, f"delta_sensitivity_aggregated_{mode}.csv")
    aggregated.to_csv(agg_csv, index=False)
    print(f"\nAggregated results (with bootstrap 95% CIs) saved to: {agg_csv}")

    print("\n" + "=" * 80)
    print(f"DELTA SENSITIVITY RESULTS ({mode.upper()} MODE, {len(seeds)} SEEDS)")
    print("=" * 80)

    for env in envs:
        print(f"\nENVIRONMENT: {env.upper()}")
        print("-" * 130)
        print(f"{'Delta':<7} | {'Mean Reward [95% CI]':<28} | {'FZC (%)':<12} | "
              f"{'Reversal Freq':<14} | {'RoC Var':<10} | {'Latency (eps, recovered only)':<30} | {'Recovery Rate':<14}")
        print("-" * 130)

        env_df = aggregated[aggregated["environment"] == env]
        for _, row in env_df.iterrows():
            fzc_str = f"{row['fzc_mean']:.1f}%" if not np.isnan(row['fzc_mean']) else "N/A"
            lat_str = f"{row['latency_mean']:.1f}+/-{row['latency_std']:.1f}" if not np.isnan(row['latency_mean']) else "N/A (0 recovered)"
            mr_str = f"{row['mean_reward']:.1f} [{row['mean_reward_ci95_lo']:.1f},{row['mean_reward_ci95_hi']:.1f}]"
            if "recovery_rate" in row and not np.isnan(row["recovery_rate"]):
                rec_str = f"{row['n_recovered']:.0f}/{row['n_recovered']+row['n_never_recovered']:.0f}"
                if row["recovery_rate"] < 1.0:
                    rec_str += "  <-- PARTIAL"
            else:
                rec_str = "N/A"
            print(f"{row['delta']:<7.2f} | {mr_str:<28} | {fzc_str:<12} | "
                  f"{row['reversal_freq_mean']:<14.4f} | {row['rate_of_change_var_mean']:<10.6f} | {lat_str:<30} | {rec_str}")
        print("-" * 130)

    for env in envs:
        env_df = aggregated[aggregated["environment"] == env]
        if not env_df.empty:
            plot_path = os.path.join(base_log_dir, f"delta_sensitivity_{env}_for_plotting_{mode}.csv")
            env_df.to_csv(plot_path, index=False)
            print(f"   Plotting data for {env}: {plot_path}")

    raw_csv = os.path.join(base_log_dir, f"delta_sensitivity_raw_seeds_{mode}.csv")
    df.to_csv(raw_csv, index=False)
    print(f"   Raw seed data: {raw_csv}")

    print("\n" + "=" * 80)
    print(f"{mode.capitalize()}-mode delta sensitivity analysis complete!")
    print(f"   Results saved to: {base_log_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="spike", choices=["steady", "spike"],
                        help="steady = no spike injection, spike = inject at episode 250")
    parser.add_argument("--envs", nargs="+", default=ENVIRONMENTS, choices=ENVIRONMENTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--deltas", nargs="+", type=float, default=DELTAS)
    args = parser.parse_args()
    main(mode=args.mode, envs=args.envs, seeds=args.seeds, deltas=args.deltas)