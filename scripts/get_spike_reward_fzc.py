import numpy as np
import os

try:
    from training.train_multi_env import FLOW_ZONES as _FLOW_ZONES_RAW
    FLOW_ZONES = {env: bounds["default"] for env, bounds in _FLOW_ZONES_RAW.items()}
except ImportError:
    FLOW_ZONES = {
        "single": (296.0, 765.0),
        "multi": (245.6, 470.1),
        "obstacle_minigrid": (-74.5, -23.7),
    }

ALGOS = ["dqn", "ddqn", "ppo"]
ENVS = ["single", "multi", "obstacle_minigrid"]
SEEDS = [42, 105, 777, 999, 2026]
WINDOW_SIZE = 10

BASE_LOG_DIRS = {
    "single": "./logs/experiments",
    "multi": "./logs/experiments",
    "obstacle_minigrid": "./logs/procedural_env_v2",
}


def run_folder_path(env, algo, seed, spike=True):
    suffix = "_spike" if spike else ""
    folder = f"{env}_{algo}_delta0p05_seed{seed}{suffix}"
    return os.path.join(BASE_LOG_DIRS[env], folder)


def compute_fzc(rolling_avgs, flow_low, flow_high, window_size=WINDOW_SIZE):
    if len(rolling_avgs) < window_size:
        return 0.0
    valid_avgs = rolling_avgs[window_size - 1:]
    in_zone = np.sum((valid_avgs >= flow_low) & (valid_avgs <= flow_high))
    return (in_zone / len(valid_avgs)) * 100.0


print(f"{'Env':<20} {'Algo':<6} | {'Spike Mean Reward':<20} | {'Spike FZC':<12}")
print("-" * 65)

for env in ENVS:
    flow_low, flow_high = FLOW_ZONES[env]
    for algo in ALGOS:
        rewards_all = []
        fzcs_all = []
        for seed in SEEDS:
            path = run_folder_path(env, algo, seed, spike=True)
            rewards_file = os.path.join(path, "episode_rewards.npy")
            rolling_file = os.path.join(path, "rolling_avg.npy")
            if not (os.path.exists(rewards_file) and os.path.exists(rolling_file)):
                print(f"  MISSING: {path}")
                continue
            rewards = np.load(rewards_file)
            rolling = np.load(rolling_file)
            rewards_all.append(float(np.mean(rewards)))
            fzcs_all.append(compute_fzc(rolling, flow_low, flow_high))

        if rewards_all:
            print(f"{env:<20} {algo.upper():<6} | {np.mean(rewards_all):>8.1f} +/- {np.std(rewards_all):<7.1f} | "
                  f"{np.mean(fzcs_all):>6.1f}% +/- {np.std(fzcs_all):<5.1f}%")