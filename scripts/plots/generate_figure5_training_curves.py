# scripts/generate_figure5_training_curves.py
"""
Figure 5 (Priority 5) -- Multi-environment training curves.

Supervisor's "Add Multi-environment training curves" is read as: training
curves across ALL THREE environments (Single, Multi, ObstacleMiniGrid),
not literally just the env internally named "Multi" -- see chat discussion.
Ancestor report's Figure 2 already showed DQN/DDQN/PPO on Single only;
this figure is meant to extend that to all three, 9 (env, algo) combos.

WHY THIS ISN'T A NAIVE "GLOB THE FIRST EVENT FILE" SCRIPT:
Per PROJECT_HANDOFF_SUMMARY §8 and user's own confirmation, run folders
can contain many TensorBoard event files -- some genuine continuations
(e.g. Procedural's confirmed 2-file 0-6M / 6M-10M split), some aborted/
restarted stubs with only a handful of logged points before a crash.
Blindly reading "the" event file, or concatenating all of them naively,
will either miss data or double-count/corrupt the curve. This script:

  1. Reads EVERY event file found under a run directory.
  2. Discards any file with fewer than MIN_POINTS_THRESHOLD scalar points
     for the target tag -- the signature of an aborted run that crashed
     almost immediately.
  3. Orders surviving files by their earliest wall-clock time (i.e.
     chronological run order: which one started first).
  4. Merges all (step -> value) pairs across files, with LATER files
     (in wall-clock order) overwriting EARLIER files at the same step --
     a resumed/restarted run re-logging a step supersedes whatever an
     earlier aborted attempt logged at that step.
  5. Sorts the merged result by step and applies rolling-mean smoothing.

Before trusting this for real: run with --inspect first (see below) to
print exactly which files were kept/discarded and why, per run folder --
verify that against what you know about your own training history before
generating the actual figure. Don't skip this -- the aborted-vs-genuine
distinction is a judgement call this script is making on your data, and
per project standard, it should be checked, not assumed.

REQUIRES: pip install tensorboard --break-system-packages   (if not already
present -- Stable-Baselines3 pulls this in for --tensorboard_log support,
so it's very likely already installed in your venv311)

Usage:
    # First, sanity-check what the script sees before generating anything:
    python -m scripts.generate_figure5_training_curves --inspect

    # Then generate the actual 3x3 figure:
    python -m scripts.generate_figure5_training_curves
"""

import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── Configuration -- FILL IN / VERIFY BEFORE RUNNING ──────────────────────── #

TAG = "rollout/ep_rew_mean"  # SB3 default logged tag. If your run_dda_only()
                              # / training script customized the logger, this
                              # will silently return zero points for every
                              # run and you'll see it immediately in --inspect
                              # output -- fix this constant if so.

MIN_POINTS_THRESHOLD = 20    # files with fewer scalar points than this for
                              # TAG are treated as aborted stubs and discarded.
                              # Tune down if a genuine short run is being
                              # wrongly discarded, tune up if a stub with a
                              # few dozen points is slipping through.

ALGOS = ["dqn", "ddqn", "ppo"]
ENVS = ["single", "multi", "obstacle_minigrid"]

# Confirmed pattern for Procedural (obstacle_minigrid), from user:
#   logs/procedural_env_v2/obstacle_minigrid_{model}_delta0p05_seed42/
#     obstacle_minigrid_{model}_delta0p05_seed42_1/   <- contains 2 event
#                                                         files (0-6M, 6M-10M)
def procedural_run_dir(algo):
    name = f"obstacle_minigrid_{algo}_delta0p05_seed42"
    return os.path.join("logs", "procedural_env_v2", name, f"{name}_1")

# Confirmed final run-attempt folder per algo (user-provided; the trailing
# _N is SB3's auto-suffix when a tb_log_name collides with a prior attempt
# -- e.g. single_ddqn hit _10, meaning 9 earlier attempts exist as SIBLING
# folders and were deliberately NOT selected here. Only the given folder
# is read; earlier-numbered sibling folders are intentionally ignored, not
# merged in -- they're superseded attempts, not continuations).
SINGLE_RUN_SUBDIR = {
    "ddqn": "single_ddqn_delta0p05_seed42_10",
    "dqn":  "single_dqn_delta0p05_seed42_3",
    "ppo":  "single_ppo_delta0p05_seed42_1",
}
MULTI_RUN_SUBDIR = {
    "ddqn": "multi_ddqn_delta0p05_seed42_6",
    "dqn":  "multi_dqn_delta0p05_seed42_6",
    "ppo":  "multi_ppo_delta0p05_seed42_2",
}

def single_run_dir(algo):
    parent = f"single_{algo}_delta0p05_seed42"
    return os.path.join("logs", "experiments", algo, parent, SINGLE_RUN_SUBDIR[algo])

def multi_run_dir(algo):
    parent = f"multi_{algo}_delta0p05_seed42"
    return os.path.join("logs", "experiments", algo, parent, MULTI_RUN_SUBDIR[algo])

RUN_DIR_RESOLVERS = {
    "single": single_run_dir,
    "multi": multi_run_dir,
    "obstacle_minigrid": procedural_run_dir,
}


# ── Core loading / merging logic ───────────────────────────────────────────── #

def load_scalar_series(event_file, tag):
    """Returns list of (step, value, wall_time) for one event file, one tag."""
    ea = EventAccumulator(event_file, size_guidance={"scalars": 0})  # 0 = load all
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return [], ea.Tags().get("scalars", [])
    events = ea.Scalars(tag)
    return [(e.step, e.value, e.wall_time) for e in events], ea.Tags()["scalars"]


def merge_run_dir(run_dir, tag, min_points, verbose=False):
    """
    Implements the discard-aborted / chronological-merge logic described in
    the module docstring. Returns (steps, values) as sorted numpy arrays,
    or (None, None) if nothing usable was found.
    """
    event_files = sorted(glob.glob(os.path.join(run_dir, "**", "events.out.tfevents*"),
                                    recursive=True))
    if not event_files:
        # also check directly inside run_dir (no subfolder nesting)
        event_files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents*")))

    if not event_files:
        if verbose:
            print(f"    No event files found under {run_dir}")
        return None, None

    file_series = []  # list of (earliest_wall_time, {step: value})
    seen_tags = set()
    for f in event_files:
        series, tags_in_file = load_scalar_series(f, tag)
        seen_tags.update(tags_in_file)
        n_points = len(series)
        keep = n_points >= min_points
        status = "KEPT" if keep else "DISCARDED (aborted stub)"
        if verbose:
            print(f"    {os.path.basename(f)}: {n_points} points -- {status}")
        if keep:
            earliest_wt = min(s[2] for s in series)
            step_val = {s[0]: s[1] for s in series}
            file_series.append((earliest_wt, step_val))

    if not file_series:
        if verbose:
            print(f"    No file in {run_dir} had >= {min_points} points for tag '{tag}'.")
            print(f"    Tags actually present in these files: {sorted(seen_tags)}")
        return None, None

    # chronological order: earliest-started file first
    file_series.sort(key=lambda x: x[0])

    merged = {}
    for _, step_val in file_series:
        merged.update(step_val)  # later (in chronological order) overwrites earlier

    steps = np.array(sorted(merged.keys()))
    values = np.array([merged[s] for s in steps])
    return steps, values


def smooth(values, window=20):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


# ── Main ────────────────────────────────────────────────────────────────── #

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inspect", action="store_true",
                     help="Print per-file keep/discard decisions for every run "
                          "without generating the figure. Run this FIRST.")
    ap.add_argument("--smooth_window", type=int, default=20)
    ap.add_argument("--output", default="./figures/figure5_training_curves.png")
    args = ap.parse_args()

    results = {}  # (env, algo) -> (steps, values)

    for env in ENVS:
        resolver = RUN_DIR_RESOLVERS[env]
        for algo in ALGOS:
            run_dir = resolver(algo)
            print(f"\n[{env}/{algo}]")
            if run_dir is None:
                print("    SKIPPED -- run_dir not configured yet (see TODO in script).")
                continue
            if not os.path.isdir(run_dir):
                print(f"    SKIPPED -- directory does not exist: {run_dir}")
                continue
            steps, values = merge_run_dir(run_dir, TAG, MIN_POINTS_THRESHOLD, verbose=True)
            if steps is None:
                print("    SKIPPED -- no usable data.")
                continue
            print(f"    Merged: {len(steps)} points, step range [{steps[0]}, {steps[-1]}]")
            results[(env, algo)] = (steps, values)

    if args.inspect:
        print("\n--inspect mode: no figure generated. Review the above, fix "
              "single_run_dir()/multi_run_dir() or MIN_POINTS_THRESHOLD if "
              "anything looks wrong, then re-run without --inspect.")
        return

    if not results:
        print("\nNo data loaded for any (env, algo) combo -- nothing to plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)
    colors = {"dqn": "#c0392b", "ddqn": "#1f4e8c", "ppo": "#3a8c4f"}
    titles = {"single": "Single", "multi": "Multi", "obstacle_minigrid": "ObstacleMiniGrid (Procedural)"}

    for ax, env in zip(axes, ENVS):
        any_plotted = False
        for algo in ALGOS:
            if (env, algo) not in results:
                continue
            steps, values = results[(env, algo)]
            sm = smooth(values, args.smooth_window)
            sm_steps = steps[len(steps) - len(sm):]
            ax.plot(sm_steps, sm, label=algo.upper(), color=colors[algo], linewidth=1.4)
            any_plotted = True
        ax.set_title(titles[env])
        ax.set_xlabel("Training Timestep")
        if any_plotted:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    axes[0].set_ylabel("Episode Reward (smoothed)")
    fig.suptitle("Figure 5 --- Training Curves Across Algorithms and Environments", fontsize=12)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=300)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()