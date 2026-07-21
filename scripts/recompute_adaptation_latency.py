"""
scripts/recompute_adaptation_latency.py
========================================
Applies the Priority 6 sustained-recovery fix to adaptation_latency.npy
for EVERY already-completed spike-mode run, WITHOUT re-running any model
evaluation.

Why this is possible without re-running anything
--------------------------------------------------
adaptation_latency was originally computed live, inside run_dda_only(),
using single-crossing detection -- but everything needed to recompute it
with the corrected sustained-crossing logic is already sitting on disk:
  - rolling_avg.npy has the full per-episode rolling-average history
  - the spike episode is always 250 (SPIKE_EPISODE, fixed across this
    project) and the window_size is always 10 (DDAController's default)
  - pre_spike_peak is reconstructible EXACTLY as the original live code
    computed it: max(rolling_avg_history[-10:]) evaluated at the moment
    episode_count == spike_episode

So this script re-derives the recovery point directly from cached data,
using the identical formula now in train_multi_env.py's run_dda_only(),
and overwrites adaptation_latency.npy in place. Verified against real
data before being written as a script: reconstructing pre_spike_peak
this way reproduced the ORIGINAL (single-crossing) adaptation_latency.npy
values exactly, confirming the reconstruction is faithful before the
sustained-crossing change is applied on top of it.

Usage:
    python -m scripts.recompute_adaptation_latency --root ./logs/experiments
    python -m scripts.recompute_adaptation_latency --root ./logs/procedural_env_v2
    python -m scripts.recompute_adaptation_latency --root ./logs/delta_sensitivity_spike
    python -m scripts.recompute_adaptation_latency --root ./logs/experiments --n 1
        # --n 1 reproduces the OLD single-crossing behavior exactly --
        # useful to confirm this script's reconstruction matches the
        # original values before trusting the sustained (--n 5) output.
"""
import os
import argparse
import numpy as np

SPIKE_EPISODE = 250
WINDOW_SIZE = 10


def recompute_one(run_dir, n_consecutive_recovery=5):
    rolling_avg_path = os.path.join(run_dir, "rolling_avg.npy")
    if not os.path.isfile(rolling_avg_path):
        return None, "no rolling_avg.npy"

    rolling_avg = np.load(rolling_avg_path)
    if len(rolling_avg) < SPIKE_EPISODE:
        return None, f"only {len(rolling_avg)} episodes, spike never reached"

    # Reconstruct pre_spike_peak EXACTLY as the original live code did:
    # dda.rolling_avg_history[-10:] evaluated the instant episode_count
    # first equals SPIKE_EPISODE (i.e. the 10 most recent entries UP TO
    # AND INCLUDING that episode).
    pre_spike_peak = float(np.max(rolling_avg[:SPIKE_EPISODE][-WINDOW_SIZE:]))
    recovery_threshold = pre_spike_peak * 0.80

    recovery_check_start = SPIKE_EPISODE + WINDOW_SIZE
    t_recovery = None
    consecutive_above = 0
    streak_start = None

    for ep in range(recovery_check_start, len(rolling_avg) + 1):
        val = rolling_avg[ep - 1]  # rolling_avg is 0-indexed, episodes are 1-indexed
        if val >= recovery_threshold:
            if consecutive_above == 0:
                streak_start = ep
            consecutive_above += 1
            if consecutive_above >= n_consecutive_recovery:
                t_recovery = streak_start
                break
        else:
            consecutive_above = 0
            streak_start = None

    if t_recovery is None:
        # Never recovered within the 500-episode run -- matches the
        # original code's fallback (adaptation_latency = n_episodes).
        latency = len(rolling_avg)
    else:
        latency = t_recovery - SPIKE_EPISODE

    return latency, None


def main(root, n_consecutive_recovery, dry_run):
    if not os.path.isdir(root):
        print(f"Root directory not found: {root}")
        return

    spike_dirs = sorted(
        os.path.join(root, d) for d in os.listdir(root)
        if "_spike" in d and os.path.isdir(os.path.join(root, d))
    )

    if not spike_dirs:
        print(f"No *_spike run folders found under {root}")
        return

    print(f"{'='*70}")
    print(f"RECOMPUTING ADAPTATION LATENCY  (n_consecutive_recovery={n_consecutive_recovery})")
    print(f"Root: {root}  |  {len(spike_dirs)} spike runs found")
    print(f"{'='*70}\n")

    changed, unchanged, skipped = 0, 0, 0

    for run_dir in spike_dirs:
        name = os.path.basename(run_dir)
        old_path = os.path.join(run_dir, "adaptation_latency.npy")
        old_val = None
        if os.path.isfile(old_path):
            old_val = float(np.load(old_path)[0])

        new_val, err = recompute_one(run_dir, n_consecutive_recovery)

        if err is not None:
            print(f"  SKIP  {name:55s} -- {err}")
            skipped += 1
            continue

        marker = ""
        if old_val is not None and abs(old_val - new_val) < 1e-6:
            unchanged += 1
        else:
            changed += 1
            marker = "  <-- CHANGED"

        print(f"  {name:55s} old={old_val}  new={new_val}{marker}")

        if not dry_run:
            np.save(old_path, np.array([new_val]))

    print(f"\n{'='*70}")
    print(f"Changed: {changed}  |  Unchanged: {unchanged}  |  Skipped: {skipped}")
    if dry_run:
        print("DRY RUN -- no files were written. Re-run without --dry_run to apply.")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                         help="Directory containing *_spike run folders "
                              "(e.g. ./logs/experiments, ./logs/procedural_env_v2, "
                              "./logs/delta_sensitivity_spike)")
    parser.add_argument("--n", type=int, default=5, dest="n_consecutive_recovery",
                         help="Consecutive episodes required above threshold. "
                              "Use --n 1 to reproduce the OLD single-crossing "
                              "behavior exactly (for verification).")
    parser.add_argument("--dry_run", action="store_true",
                         help="Print what would change without writing any files.")
    args = parser.parse_args()
    main(args.root, args.n_consecutive_recovery, args.dry_run)