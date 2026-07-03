# scripts/run_all_evaluations.py
"""
Automated batch evaluation script.
Runs all 3 trained models across both environments over 5 random seeds.

Modes:
    --mode steady : No spike protocol (max_speed=9.0) — steady-state regulation
    --mode spike  : With spike protocol (max_speed=9.0) — stress-test evaluation

Total runs: 3 * 2 * 5 = 30 runs (500 episodes each) per mode.
"""
import os
import sys
import argparse
import subprocess

# Define experimental scope parameters
SEEDS = [42, 105, 777, 999, 2026]
ALGOS = ["dqn", "ddqn", "ppo"]
ENVS = ["single", "multi"]
DELTA = 0.05
MAX_SPEED = 9.0  # Unified speed ceiling for both modes
BASE_LOG_DIR = "./logs/experiments"

def main(mode="spike"):
    print("=" * 70)
    print(f"STARTING AUTOMATED BATCH EVALUATION ({mode.upper()} MODE)")
    print("=" * 70)

    # Determine spike episode based on mode
    if mode == "steady":
        spike_episode = None  # No spike
        mode_suffix = "steadystate"
    else:  # spike
        spike_episode = 250   # Spike at episode 250
        mode_suffix = "spike"

    print(f"  Mode       : {mode.upper()}")
    print(f"  Max Speed  : {MAX_SPEED} (unified across modes)")
    print(f"  Spike      : {'Enabled at ep ' + str(spike_episode) if spike_episode else 'Disabled'}")
    print(f"  Seeds      : {SEEDS}")
    print(f"  Algorithms : {ALGOS}")
    print(f"  Environments: {ENVS}")
    print("=" * 70)

    run_idx = 0
    failed_runs = 0

    for env in ENVS:
        for algo in ALGOS:
            # Training runs defaulted to delta=0.05 and seed=42
            model_folder = f"{env}_{algo}_delta0p05_seed42"
            model_path = os.path.join(BASE_LOG_DIR, model_folder, "best_model.zip")
            
            if not os.path.exists(model_path):
                print(f"⚠️ Skipping: Checkpoint not found at {model_path}")
                continue
                
            for seed in SEEDS:
                run_idx += 1
                print(f"\n🚀 [{run_idx}/30] Evaluating: Env={env.upper()} | Algo={algo.upper()} | Seed={seed} | Mode={mode.upper()}")

                # Build the command
                cmd = [
                    sys.executable, "-m", "training.train_multi_env",
                    "--algo", algo,
                    "--env", env,
                    "--resume", model_path,
                    "--delta", str(DELTA),
                    "--seed", str(seed),
                    "--max_speed", str(MAX_SPEED),
                ]

                # Add spike flag if in spike mode
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
        print(f"✅ SUCCESS: All 30 multi-seed DDA evaluations ({mode.upper()} mode) completed successfully!")
        print(f"All arrays saved to individual run folders under: {BASE_LOG_DIR}")
    else:
        print(f"⚠️ Warning: Batch execution finished with {failed_runs} failed runs. Check errors above.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch DDA evaluations")
    parser.add_argument("--mode", type=str, default="spike", choices=["steady", "spike"],
                        help="Evaluation mode: 'steady' (no spike) or 'spike' (with recovery test)")
    args = parser.parse_args()
    main(mode=args.mode)