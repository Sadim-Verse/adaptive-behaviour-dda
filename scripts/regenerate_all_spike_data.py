import subprocess
import sys

SEEDS = [42, 105, 777, 999, 2026]
ALGOS = ["dqn", "ddqn", "ppo"]
ENVS = ["single", "multi", "obstacle_minigrid"]

MODEL_PATHS = {
    "single":            "./logs/experiments/{algo}/single_{algo}_delta0p05_seed42/best_model.zip",
    "multi":              "./logs/experiments/{algo}/multi_{algo}_delta0p05_seed42/best_model.zip",
    "obstacle_minigrid":   "./logs/procedural_env_v2/obstacle_minigrid_{algo}_delta0p05_seed42/best_model.zip",
}
OUTPUT_DIRS = {
    "single": "./logs/experiments",
    "multi": "./logs/experiments",
    "obstacle_minigrid": "./logs/procedural_env_v2",
}
MAX_SPEED = {"single": 9.0, "multi": 9.0, "obstacle_minigrid": 1.0}

print("=" * 80)
print("PART 1: Tables 4 / 8 -- all 3 algorithms, delta=0.05, all 3 environments, spike mode")
print("=" * 80)

for env in ENVS:
    for algo in ALGOS:
        model_path = MODEL_PATHS[env].format(algo=algo)
        for seed in SEEDS:
            cmd = [
                sys.executable, "-m", "training.train_multi_env",
                "--algo", algo, "--env", env, "--resume", model_path,
                "--delta", "0.05", "--seed", str(seed),
                "--max_speed", str(MAX_SPEED[env]),
                "--spike", "250",
                "--output", OUTPUT_DIRS[env],
            ]
            print(f"\n>>> {env} / {algo} / seed {seed}")
            subprocess.run(cmd, check=True)

print("\n" + "=" * 80)
print("PART 2: Table 5 -- no_rolling_window / no_reward_smoothing ablations, spike mode, all 3 envs")
print("=" * 80)

ABLATION_MODEL_PATHS = {
    "single": "./logs/experiments/ddqn/single_ddqn_delta0p05_seed42/best_model.zip",
    "multi": "./logs/experiments/ddqn/multi_ddqn_delta0p05_seed42/best_model.zip",
    "obstacle_minigrid": "./logs/procedural_env_v2/obstacle_minigrid_ddqn_delta0p05_seed42/best_model.zip",
}
ABLATIONS = {"no_rolling_window": "no_rolling_window", "no_reward_smoothing": "no_smoothing"}

for env in ENVS:
    for ablation_name, ablation_flag in ABLATIONS.items():
        model_path = ABLATION_MODEL_PATHS[env]
        for seed in SEEDS:
            cmd = [
                sys.executable, "-m", "training.train_multi_env",
                "--algo", "ddqn", "--env", env, "--resume", model_path,
                "--delta", "0.05", "--seed", str(seed),
                "--max_speed", "9.0",
                "--spike", "250",
                "--ablation", ablation_flag,
                "--output", f"./logs/ablations/{ablation_name}",
            ]
            print(f"\n>>> {env} / {ablation_name} / seed {seed}")
            subprocess.run(cmd, check=True)

print("\nAll regeneration complete.")