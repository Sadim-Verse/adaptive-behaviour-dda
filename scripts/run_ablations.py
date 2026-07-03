# scripts/run_ablations.py
"""
Run all ablation studies for Table 5.
- 4 ablation configurations: no_dda, fixed_difficulty, no_rolling_window, no_reward_smoothing
- 2 environments: single, multi
- 5 seeds: 42, 105, 777, 999, 2026
- 1 algorithm: DDQN (the primary agent)

Total runs: 4 ablations × 2 envs × 5 seeds = 40 runs (~2 hours)

Results saved to: ./logs/ablations/
Folder naming: {env}_ddqn_abl_{ablation_name}_seed{seed}
"""

import os
import sys
import subprocess
import time

# ── Configuration ─────────────────────────────────────────────────────────── #

SEEDS = [42, 105, 777, 999, 2026]
ENVIRONMENTS = ["single", "multi"]
BASE_LOG_DIR = "./logs/ablations"

# Challenge speed for "No DDA" ablation — chosen to challenge the robust DDQN agent
CHALLENGE_SPEED = 6.0

# Path to trained DDQN model
MODEL_PATH = {
    "single": "./logs/experiments/ddqn/single_ddqn_delta0p05_seed42/best_model.zip",
    "multi": "./logs/experiments/ddqn/multi_ddqn_delta0p05_seed42/best_model.zip"
}

# ── Ablation Configurations ──────────────────────────────────────────────── #

ABLATIONS = {
    "no_dda": {
        "description": f"No DDA Controller — fixed speed at challenge level ({CHALLENGE_SPEED})",
        "cmd_args": [
            "--max_speed", str(CHALLENGE_SPEED)
        ],
        "ablation_flag": "no_dda"
    },
    "fixed_difficulty": {
        "description": "DDA Controller exists but holds speed constant",
        "cmd_args": [
            "--max_speed", "9.0"
        ],
        "ablation_flag": "fixed_difficulty"
    },
    "no_rolling_window": {
        "description": "DDA uses immediate reward instead of rolling average",
        "cmd_args": [
            "--max_speed", "9.0"
        ],
        "ablation_flag": "no_rolling_window"
    },
    "no_reward_smoothing": {
        "description": "DDA uses raw rewards without any smoothing",
        "cmd_args": [
            "--max_speed", "9.0"
        ],
        "ablation_flag": "no_smoothing"
    }
}

# ── Helper ────────────────────────────────────────────────────────────────── #

def build_run_folder(ablation_name, env, seed):
    """
    Build the folder name as ACTUALLY created by train_multi_env.py's main().
    Pattern: {env}_{algo}_delta{delta}_seed{seed}_abl_{ablation}
    e.g.     single_ddqn_delta0p05_seed42_abl_fixed_difficulty
    (Algorithm is always ddqn here, delta is always 0.05.)
    """
    return f"{env}_ddqn_delta0p05_seed{seed}_abl_{ablation_name}"

# ── Main ─────────────────────────────────────────────────────────────────── #

def run_ablation(ablation_name, env, seed):
    """Run a single ablation run."""
    model_path = MODEL_PATH[env]
    run_folder = build_run_folder(ablation_name, env, seed)
    ablation_config = ABLATIONS[ablation_name]
    
    cmd = [
        sys.executable, "-m", "training.train_multi_env",
        "--algo", "ddqn",
        "--env", env,
        "--resume", model_path,
        "--delta", "0.05",
        "--seed", str(seed),
        "--output", os.path.join(BASE_LOG_DIR, ablation_name)
    ]
    
    cmd.extend(ablation_config["cmd_args"])
    
    if ablation_config["ablation_flag"] is not None:
        cmd.extend(["--ablation", ablation_config["ablation_flag"]])
    
    print(f"  Running: {' '.join(cmd)}")
    print(f"  Saving to: {run_folder}")
    
    start_time = time.time()
    try:
        subprocess.run(cmd, check=True, capture_output=False)
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed: {e}")
        return False
    wall_time = time.time() - start_time
    print(f"  ✅ Completed in {wall_time:.1f}s")
    return True

def main():
    print("=" * 80)
    print("ABLATION STUDIES")
    print(f"  Ablations: {', '.join(ABLATIONS.keys())}")
    print(f"  Environments: {ENVIRONMENTS}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Challenge Speed (No DDA): {CHALLENGE_SPEED}")
    print(f"  Total runs: {len(ABLATIONS) * len(ENVIRONMENTS) * len(SEEDS)}")
    print("=" * 80)

    failed = 0
    total = 0

    for ablation_name, ablation_config in ABLATIONS.items():
        print(f"\n{'─'*80}")
        print(f"ABLATION: {ablation_name.upper()}")
        print(f"  {ablation_config['description']}")
        if ablation_config["ablation_flag"] is not None:
            print(f"  --ablation flag: {ablation_config['ablation_flag']}")
        else:
            print(f"  --ablation flag: None (fixed speed only)")
        print(f"{'─'*80}")

        for env in ENVIRONMENTS:
            print(f"\n  Environment: {env.upper()}")

            for seed in SEEDS:
                total += 1
                print(f"\n    Seed {seed}...")
                if run_ablation(ablation_name, env, seed):
                    print(f"      ✅ Success")
                else:
                    print(f"      ❌ Failed")
                    failed += 1

    print("\n" + "=" * 80)
    print(f"✅ Ablation studies complete!")
    print(f"  Total runs: {total}")
    print(f"  Failed: {failed}")
    print(f"  Results saved to: {BASE_LOG_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()