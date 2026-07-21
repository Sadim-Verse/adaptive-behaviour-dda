# scripts/run_all_evaluations.py
"""
Automated batch evaluation script.
Runs all 3 trained models across all 3 environments over 5 evaluation seeds.

Modes:
    --mode steady : No spike protocol — steady-state regulation
    --mode spike  : With spike protocol — stress-test evaluation

Total runs: 3 algos * 3 envs * 5 seeds = 45 runs (500 episodes each) per mode.

NOTE on max_speed: single/multi use a physical speed scale (9.0 = the
project's existing unified ceiling for both modes). obstacle_minigrid uses
a NORMALISED [0,1] difficulty scale instead (see DDA_SCALE in
train_multi_env.py) -- passing 9.0 to it would be meaningless (its
set_difficulty() clips to [0,1] anyway, silently saturating at 1.0 instead
of erroring, which would be a much harder bug to notice than an explicit
per-environment value here).
"""
import os
import sys
import argparse
import subprocess

from training.train_multi_env import DDA_SCALE

# Define experimental scope parameters
SEEDS = [42, 105, 777, 999, 2026]
ALGOS = ["dqn", "ddqn", "ppo"]
ENVS = ["single", "multi", "obstacle_minigrid"]
DELTA = 0.05
# Same folder-layout reality as calibrate_flow_zones.py: obstacle_minigrid
# runs live under a different --output folder than single/multi did.
BASE_LOG_DIRS = {
    "single": "./logs/experiments",
    "multi": "./logs/experiments",
    "obstacle_minigrid": "./logs/procedural_env_v2",
}

# Per-environment max_speed (DDA ceiling during evaluation). single/multi
# keep the project's existing unified value; obstacle_minigrid gets its
# own scale's ceiling instead of inheriting 9.0.
MAX_SPEED = {
    "single": 9.0,
    "multi": 9.0,
    "obstacle_minigrid": DDA_SCALE["obstacle_minigrid"]["max_speed_default"],  # 1.0
}


def main(mode="spike", envs=None, algos=None, seeds=None):
    envs = envs or ENVS
    algos = algos or ALGOS
    seeds = seeds or SEEDS
    print("=" * 70)
    print(f"STARTING AUTOMATED BATCH EVALUATION ({mode.upper()} MODE)")
    print("=" * 70)

    if mode == "steady":
        spike_episode = None
    else:
        spike_episode = 250

    total_runs = len(envs) * len(algos) * len(seeds)
    print(f"  Mode        : {mode.upper()}")
    print(f"  Spike       : {'Enabled at ep ' + str(spike_episode) if spike_episode else 'Disabled'}")
    print(f"  Seeds       : {seeds}")
    print(f"  Algorithms  : {algos}")
    print(f"  Environments: {envs}")
    print(f"  Total runs  : {total_runs}")
    print("=" * 70)

    run_idx = 0
    failed_runs = 0

    for env in envs:
        max_speed = MAX_SPEED[env]
        for algo in algos:
            model_folder = f"{env}_{algo}_delta0p05_seed42"
            # single/multi have an extra {algo}-named subfolder that
            # obstacle_minigrid doesn't -- see calibrate_flow_zones.py's
            # model_path_for() docstring for the same real inconsistency.
            if env in ("single", "multi"):
                model_path = os.path.join(BASE_LOG_DIRS[env], algo, model_folder, "best_model.zip")
            else:
                model_path = os.path.join(BASE_LOG_DIRS[env], model_folder, "best_model.zip")

            if not os.path.exists(model_path):
                print(f"⚠️ Skipping: Checkpoint not found at {model_path}")
                continue

            for seed in seeds:
                run_idx += 1
                print(f"\n🚀 [{run_idx}/{total_runs}] Evaluating: Env={env.upper()} | "
                      f"Algo={algo.upper()} | Seed={seed} | Mode={mode.upper()} | max_speed={max_speed}")

                cmd = [
                    sys.executable, "-m", "training.train_multi_env",
                    "--algo", algo,
                    "--env", env,
                    "--resume", model_path,
                    "--delta", str(DELTA),
                    "--seed", str(seed),
                    "--max_speed", str(max_speed),
                    "--output", BASE_LOG_DIRS[env],
                ]

                if mode == "spike":
                    cmd.append("--spike")
                    cmd.append(str(spike_episode))

                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"❌ Error occurred while executing evaluation for seed {seed}: {e}")
                    failed_runs += 1
                    continue

    print("\n" + "=" * 70)
    if failed_runs == 0:
        print(f"✅ SUCCESS: All {total_runs} DDA evaluations ({mode.upper()} mode) completed successfully!")
        print(f"All arrays saved under each environment's own output folder: {BASE_LOG_DIRS}")
    else:
        print(f"⚠️ Warning: Batch execution finished with {failed_runs} failed runs. Check errors above.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch DDA evaluations")
    parser.add_argument("--mode", type=str, default="spike", choices=["steady", "spike"],
                        help="Evaluation mode: 'steady' (no spike) or 'spike' (with recovery test)")
    parser.add_argument("--envs", nargs="+", default=None,
                        choices=["single", "multi", "obstacle_minigrid"],
                        help="Which environments to evaluate (default: all three)")
    parser.add_argument("--algos", nargs="+", default=None,
                        choices=["dqn", "ddqn", "ppo"],
                        help="Which algorithms to evaluate (default: all three). "
                             "Use to smoke-test a narrow slice before the full batch.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Which evaluation seeds to use (default: all 5). "
                             "Use e.g. --seeds 42 to smoke-test with just one.")
    args = parser.parse_args()
    main(mode=args.mode, envs=args.envs, algos=args.algos, seeds=args.seeds)