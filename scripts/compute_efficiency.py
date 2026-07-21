"""
scripts/table7_compute_efficiency.py
======================================
Extracts Table 7 (Computational Efficiency) values directly from
TensorBoard event files, for any algorithm/environment combination you
already have training logs for.

Reports TWO different FPS figures, because they can legitimately differ
and your existing table's numbers don't arithmetically match a naive
total_steps / wall_clock_time calculation (e.g. DQN: 1M steps in ~30 min
= ~556 steps/sec, not ~6,000):

  1. "sb3_fps"    -- the mean of SB3's own internally-logged time/fps
                     scalar (a rolling throughput metric SB3 computes and
                     logs automatically during training). This is almost
                     certainly what the existing Single/Multi table
                     values came from.
  2. "naive_fps"  -- total_timesteps / total_wall_clock_seconds, a
                     simple recomputation from the same event file, for
                     comparison.

Run this on Single/Multi FIRST to confirm which figure matches your
existing table before trusting new Procedural numbers -- do not assume,
check.

Usage:
    python -m scripts.table7_compute_efficiency
    python -m scripts.table7_compute_efficiency --envs obstacle_minigrid
    python -m scripts.table7_compute_efficiency --envs single multi obstacle_minigrid --seed 42
"""

import os
import glob
import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ALGOS = ["dqn", "ddqn", "ppo"]
ENVS = ["single", "multi", "obstacle_minigrid"]

BASE_LOG_DIRS = {
    "single": "./logs/experiments",
    "multi": "./logs/experiments",
    "obstacle_minigrid": "./logs/procedural_env_v2",
}

# Same real inconsistency handled elsewhere in this project: single/multi
# training output lives under an extra {algo}/ subfolder, obstacle_minigrid
# does not.
def training_folder(env, algo, seed):
    folder = f"{env}_{algo}_delta0p05_seed{seed}"
    base = BASE_LOG_DIRS[env]
    if env in ("single", "multi"):
        return os.path.join(base, algo, folder)
    return os.path.join(base, folder)


def find_event_file(training_dir):
    matches = glob.glob(os.path.join(training_dir, "**", "events.out.tfevents.*"), recursive=True)
    if not matches:
        return None
    # If multiple event files exist (aborted/restarted runs -- seen before
    # in this project's Single/Multi training folders), use the LARGEST
    # one by file size as a proxy for "the real, complete run", not just
    # the first glob match.
    return max(matches, key=os.path.getsize)


def extract_metrics(event_file):
    ea = EventAccumulator(event_file, size_guidance={"scalars": 0})  # 0 = load all, don't truncate
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if not tags:
        return None

    result = {}

    # Wall-clock elapsed time: first to last event timestamp, across
    # WHICHEVER tag has the most points (most complete signal).
    best_tag = max(tags, key=lambda t: len(ea.Scalars(t)))
    events = ea.Scalars(best_tag)
    wall_start, wall_end = events[0].wall_time, events[-1].wall_time
    elapsed_seconds = wall_end - wall_start
    result["elapsed_minutes"] = elapsed_seconds / 60.0
    result["elapsed_hours"] = elapsed_seconds / 3600.0

    # Total timesteps reached (SB3 logs against global_step).
    result["total_timesteps"] = events[-1].step

    # naive FPS
    result["naive_fps"] = result["total_timesteps"] / elapsed_seconds if elapsed_seconds > 0 else float("nan")

    # SB3's own internally-logged FPS, if present
    fps_tag = next((t for t in tags if t.endswith("fps") or t == "time/fps"), None)
    if fps_tag:
        fps_events = ea.Scalars(fps_tag)
        values = [e.value for e in fps_events]
        result["sb3_fps_mean"] = sum(values) / len(values)
        result["sb3_fps_last"] = values[-1]
    else:
        result["sb3_fps_mean"] = None
        result["sb3_fps_last"] = None

    result["n_scalar_tags"] = len(tags)
    result["event_points_used"] = len(events)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="+", default=ENVS, choices=ENVS)
    ap.add_argument("--algos", nargs="+", default=ALGOS, choices=ALGOS)
    ap.add_argument("--seed", type=int, default=42,
                     help="Which seed's training run to measure (default 42, "
                          "matching the seed used throughout this project's "
                          "reported/reference results).")
    args = ap.parse_args()

    print("=" * 100)
    print("TABLE 7 -- COMPUTATIONAL EFFICIENCY EXTRACTION")
    print(f"Seed: {args.seed}")
    print("=" * 100)
    print(f"{'Env':<20} {'Algo':<6} | {'Elapsed':<12} | {'Steps':<10} | "
          f"{'naive_fps':<10} | {'sb3_fps(mean)':<14} | {'sb3_fps(last)':<14}")
    print("-" * 100)

    rows = []
    for env in args.envs:
        for algo in args.algos:
            training_dir = training_folder(env, algo, args.seed)
            if not os.path.isdir(training_dir):
                print(f"{env:<20} {algo.upper():<6} | MISSING FOLDER: {training_dir}")
                continue

            event_file = find_event_file(training_dir)
            if event_file is None:
                print(f"{env:<20} {algo.upper():<6} | NO EVENT FILE found under {training_dir}")
                continue

            m = extract_metrics(event_file)
            if m is None:
                print(f"{env:<20} {algo.upper():<6} | event file found but no scalar data: {event_file}")
                continue

            elapsed_str = (f"{m['elapsed_hours']:.2f} hr" if m['elapsed_hours'] >= 1
                            else f"{m['elapsed_minutes']:.1f} min")
            sb3_mean_str = f"{m['sb3_fps_mean']:.0f}" if m['sb3_fps_mean'] is not None else "N/A"
            sb3_last_str = f"{m['sb3_fps_last']:.0f}" if m['sb3_fps_last'] is not None else "N/A"

            print(f"{env:<20} {algo.upper():<6} | {elapsed_str:<12} | {m['total_timesteps']:<10,} | "
                  f"{m['naive_fps']:<10.1f} | {sb3_mean_str:<14} | {sb3_last_str:<14}")

            rows.append((env, algo, m, event_file))

    print("-" * 100)
    print("\nVERIFICATION STEP -- do this before trusting any new (Procedural) numbers:")
    print("  Compare this script's Single/Multi output above against your EXISTING Table 7:")
    print("    DQN  Single ~30 min, ~6,000 FPS   |  Multi ~180 min, ~6,000 FPS")
    print("    DDQN Single ~30 min, ~6,000 FPS   |  Multi ~180 min, ~6,000 FPS")
    print("    PPO  Single ~40 min, ~4,500 FPS   |  Multi ~220 min, ~4,500 FPS")
    print("  Whichever FPS column (naive_fps vs sb3_fps_mean vs sb3_fps_last) matches")
    print("  those existing numbers is the convention to use for Procedural's new row.")
    print("  If NEITHER matches closely, the existing table's FPS figures may have come")
    print("  from a different source entirely (e.g. manual observation, a different SB3")
    print("  version's logging behavior) -- flag this rather than silently picking one.")
    print("=" * 100)


if __name__ == "__main__":
    main()