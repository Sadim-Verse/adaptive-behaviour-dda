"""
scripts/audit_evaluation_data.py
==================================
Checks that every expected run (3 environments x 3 algorithms x 5 seeds x
2 modes = 90 runs) actually produced complete output, before any Track A
analysis is computed on top of it. Written specifically because the
obstacle_minigrid steady-mode batch already had one silent gap (disk-space
failure mid-run) -- this checks for that same failure mode across
EVERYTHING, not just the run that already visibly failed, since a partial
disk-space or interruption failure could in principle have hit any run
without necessarily printing a big traceback you'd notice.

Usage:
    python -m scripts.audit_evaluation_data
"""

import os
import numpy as np

SEEDS = [42, 105, 777, 999, 2026]
ALGOS = ["dqn", "ddqn", "ppo"]
ENVS = ["single", "multi", "obstacle_minigrid"]
MODES = ["steady", "spike"]

BASE_LOG_DIRS = {
    "single": "./logs/experiments",
    "multi": "./logs/experiments",
    "obstacle_minigrid": "./logs/procedural_env_v2",
}

EXPECTED_FILES_STEADY = ["episode_rewards.npy", "speed_history.npy", "rolling_avg.npy"]
EXPECTED_FILES_SPIKE = EXPECTED_FILES_STEADY + ["adaptation_latency.npy"]


def run_folder(env, algo, seed, mode):
    """
    Path to EVALUATION OUTPUT (episode_rewards.npy etc), NOT to the
    trained model checkpoint. These are genuinely different conventions
    in this project:
      - trained MODEL checkpoints: logs/experiments/{algo}/{run_id}/best_model.zip
        (single/multi only -- an extra algo-named subfolder)
      - evaluation OUTPUT (.npy files): logs/experiments/{run_id}/
        (always flat, no algo subfolder, for ALL three environments --
        because run_all_evaluations.py passes --output as just the base
        dir, and train_multi_env.py builds log_dir = output_dir/run_id
        with no extra nesting)
    Applying the model-checkpoint's algo-subfolder convention to output
    paths was a real bug -- these are two different things that happen
    to share a similar-looking base directory.
    """
    suffix = "_spike" if mode == "spike" else ""
    folder = f"{env}_{algo}_delta0p05_seed{seed}{suffix}"
    return os.path.join(BASE_LOG_DIRS[env], folder)


def main():
    missing = []
    suspicious = []
    total = 0

    for env in ENVS:
        for algo in ALGOS:
            for seed in SEEDS:
                for mode in MODES:
                    total += 1
                    folder = run_folder(env, algo, seed, mode)
                    expected = EXPECTED_FILES_SPIKE if mode == "spike" else EXPECTED_FILES_STEADY

                    if not os.path.isdir(folder):
                        missing.append((env, algo, seed, mode, "folder does not exist"))
                        continue

                    for fname in expected:
                        fpath = os.path.join(folder, fname)
                        if not os.path.exists(fpath):
                            missing.append((env, algo, seed, mode, f"missing {fname}"))
                            continue
                        # sanity check: episode_rewards.npy should have ~500 entries
                        # (a truncated/corrupted save from an interrupted run would
                        # often have fewer, or be empty)
                        if fname == "episode_rewards.npy":
                            try:
                                arr = np.load(fpath)
                                if len(arr) < 500:
                                    suspicious.append(
                                        (env, algo, seed, mode,
                                         f"episode_rewards.npy has only {len(arr)} episodes (expected 500)")
                                    )
                                if np.isnan(arr).any():
                                    suspicious.append((env, algo, seed, mode, "episode_rewards.npy contains NaN"))
                            except Exception as e:
                                missing.append((env, algo, seed, mode, f"episode_rewards.npy unreadable: {e}"))

    print("=" * 70)
    print(f"AUDIT COMPLETE: {total} expected runs checked "
          f"({len(ENVS)} envs x {len(ALGOS)} algos x {len(SEEDS)} seeds x {len(MODES)} modes)")
    print("=" * 70)

    if not missing and not suspicious:
        print("\n✅ All runs complete. No gaps found.")
    else:
        if missing:
            print(f"\n❌ {len(missing)} MISSING (need to be rerun):")
            for env, algo, seed, mode, reason in missing:
                print(f"   {env}/{algo}/seed{seed}/{mode}: {reason}")
        if suspicious:
            print(f"\n⚠️  {len(suspicious)} SUSPICIOUS (present but look wrong -- inspect before trusting):")
            for env, algo, seed, mode, reason in suspicious:
                print(f"   {env}/{algo}/seed{seed}/{mode}: {reason}")

        # print ready-to-use rerun commands grouped by env/mode/algo for
        # whatever seeds are missing, using the --algos/--seeds filters
        if missing:
            print("\n--- Suggested rerun commands ---")
            from collections import defaultdict
            groups = defaultdict(set)
            for env, algo, seed, mode, _ in missing:
                groups[(env, algo, mode)].add(seed)
            for (env, algo, mode), seeds in groups.items():
                seed_args = " ".join(str(s) for s in sorted(seeds))
                print(f"python -m scripts.run_all_evaluations --mode {mode} "
                      f"--envs {env} --algos {algo} --seeds {seed_args}")

    print("=" * 70)


if __name__ == "__main__":
    main()