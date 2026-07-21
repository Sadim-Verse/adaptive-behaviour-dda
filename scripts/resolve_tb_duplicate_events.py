"""
scripts/resolve_tb_duplicate_events.py
========================================
When a training run is resumed (extend_training.py) and SB3's logger
reuses the same tb_log_name, TensorBoard can end up with TWO separate
event files inside the same run folder (e.g.
obstacle_minigrid_dqn_delta0p05_seed42_1/events.out.tfevents.<time1>...
obstacle_minigrid_dqn_delta0p05_seed42_1/events.out.tfevents.<time2>...).
Reading scalars via the TensorBoard UI or a naive multi-file loader can
then silently merge/overlap two different histories for the same step
range, producing exactly the "same step, contradictory value" problem
found when comparing this project's 0-6M and 6M-10M exports.

This script loads each event file SEPARATELY and explicitly, using
EventAccumulator directly on individual files (not a directory glob),
so you can see exactly what each file actually contains before any
merging decision is made -- rather than trusting an automatic merge.

Usage:
    python -m scripts.resolve_tb_duplicate_events --run_dir "logs/procedural_env_v2/obstacle_minigrid_dqn_delta0p05_seed42" --tag "eval/mean_reward"
"""

import os
import glob
import argparse
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_single_event_file(filepath, tag):
    """Load scalars for one tag from exactly one event file."""
    ea = EventAccumulator(filepath, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    wall_times = np.array([e.wall_time for e in events])
    return steps, values, wall_times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to the run folder (the one containing the "
                             "events.out.tfevents.* files directly, e.g. "
                             "logs/.../obstacle_minigrid_dqn_delta0p05_seed42_1)")
    parser.add_argument("--tag", type=str, default="eval/mean_reward",
                        help="TensorBoard scalar tag to extract")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Where to save the cleaned, merged CSV (default: "
                             "<run_dir>/resolved_<tag_sanitized>.csv)")
    args = parser.parse_args()

    event_files = sorted(glob.glob(os.path.join(args.run_dir, "events.out.tfevents.*")))
    if not event_files:
        # maybe run_dir is the PARENT and the actual tfevents files are one level down
        event_files = sorted(glob.glob(os.path.join(args.run_dir, "**", "events.out.tfevents.*"),
                                        recursive=True))

    if not event_files:
        print(f"No event files found under {args.run_dir}")
        return

    print(f"Found {len(event_files)} event file(s):")
    for f in event_files:
        print(f"  {f}")
    print()

    all_data = []  # list of (steps, values, wall_times, source_file)
    for f in event_files:
        result = load_single_event_file(f, args.tag)
        if result is None:
            print(f"  [{os.path.basename(f)}] tag '{args.tag}' not found, skipping")
            continue
        steps, values, wall_times = result
        print(f"  [{os.path.basename(f)}]")
        print(f"    n_points   : {len(steps)}")
        print(f"    step range : {steps.min()} - {steps.max()}")
        print(f"    value range: [{values.min():.1f}, {values.max():.1f}]  mean={values.mean():.1f}")
        print(f"    wall time  : {wall_times.min():.0f} - {wall_times.max():.0f} "
              f"({(wall_times.max()-wall_times.min())/3600:.2f} hours)")
        all_data.append((steps, values, wall_times, f))
        print()

    if len(all_data) == 0:
        print("No usable data found in any event file.")
        return

    if len(all_data) == 1:
        print("Only one file had data -- no merge ambiguity, using it directly.")
        steps, values, wall_times, _ = all_data[0]
    else:
        print("=" * 60)
        print("MERGE STRATEGY: sort files chronologically by their EARLIEST "
              "wall_time (oldest file first). For any step covered by an "
              "earlier file, that file's value wins -- later files only "
              "contribute steps NOT already covered by an earlier one.")
        print("Rationale: a resumed run re-logging an old step is the "
              "anomaly to discard, not the original recording of that step. "
              "This is a reasonable DEFAULT, not a guarantee -- inspect the "
              "per-file stats printed above yourself before trusting this "
              "blindly, especially if the per-file value ranges look "
              "implausible for what you know about training at that stage.")
        print("=" * 60)

        # sort files by their own earliest wall_time (chronological order
        # of file CREATION, not of individual point write times)
        all_data_sorted = sorted(all_data, key=lambda d: d[2].min())

        resolved = {}  # step -> (value, wall_time, source_file)
        for steps, values, wall_times, fname in all_data_sorted:
            for s, v, w in zip(steps, values, wall_times):
                if s not in resolved:
                    resolved[s] = (v, w, fname)
                # else: a later file tried to re-supply a step an earlier
                # file already covered -- earlier file wins, skip it.

        steps = np.array(sorted(resolved.keys()))
        values = np.array([resolved[s][0] for s in steps])
        wall_times = np.array([resolved[s][1] for s in steps])

        total_raw = sum(len(d[0]) for d in all_data)
        n_duplicates = total_raw - len(steps)
        print(f"\nResolved {total_raw} raw points -> {len(steps)} unique steps "
              f"({n_duplicates} duplicate/superseded step entries discarded).")

    out_path = args.output_csv or os.path.join(
        args.run_dir, f"resolved_{args.tag.replace('/', '_')}.csv"
    )
    import csv
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Wall time", "Step", "Value"])
        for w, s, v in zip(wall_times, steps, values):
            writer.writerow([w, s, v])

    print(f"\nClean, de-duplicated CSV saved to: {out_path}")
    print(f"Final: n={len(steps)}, step range {steps.min()}-{steps.max()}, "
          f"value range [{values.min():.1f}, {values.max():.1f}]")


if __name__ == "__main__":
    main()