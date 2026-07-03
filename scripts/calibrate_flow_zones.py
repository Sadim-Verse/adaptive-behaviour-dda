"""
scripts/calibrate_flow_zones.py
================================
Empirically calibrates Flow Zone bounds for each environment
by running static (no-DDA) baseline evaluations and computing
the 30th and 70th percentiles of the pooled reward distribution.

Usage:
    python -m scripts.calibrate_flow_zones

Output:
    - Prints the calibrated bounds for each environment
    - Saves calibration data to logs/calibration/
    - Prints the exact FLOW_ZONES dict to paste into train_multi_env.py
"""

import os
import numpy as np
from stable_baselines3 import DQN, PPO
from training.ddqn import DDQN
from training.train_multi_env import make_env

# ── Configuration ─────────────────────────────────────────────────────────── #

N_EPISODES   = 100      # episodes per algorithm per environment
FIXED_SPEED  = 2.0      # baseline speed (no DDA)
BASE_LOG_DIR = "./logs/experiments"
CALIB_DIR    = "./logs/calibration"
os.makedirs(CALIB_DIR, exist_ok=True)

# Map algo name → (model class, model folder pattern)
ALGO_CONFIG = {
    "dqn":  (DQN,  "single_dqn_delta0p05_seed42",  "multi_dqn_delta0p05_seed42"),
    "ddqn": (DDQN, "single_ddqn_delta0p05_seed42", "multi_ddqn_delta0p05_seed42"),
    "ppo":  (PPO,  "single_ppo_delta0p05_seed42",  "multi_ppo_delta0p05_seed42"),
}

ENVS = ["single", "multi"]

# ── Helper ────────────────────────────────────────────────────────────────── #

def evaluate_static(model, env_type, n_episodes, fixed_speed, algo_name):
    """Run frozen policy for n_episodes at fixed speed. Returns list of rewards."""
    env = make_env(env_type=env_type, initial_speed=fixed_speed)
    rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=200 + ep)   # separate seed range from eval runs
        total = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)
        if (ep + 1) % 20 == 0:
            print(f"    [{algo_name.upper()}] Episode {ep+1:3d}/{n_episodes} "
                  f"| reward: {total:7.1f} | running mean: {np.mean(rewards):.1f}")

    env.close()
    return rewards


def compute_bounds(all_rewards, env_type):
    """
    Compute flow zone bounds from pooled reward distribution.

    Single environment: use 30% and 60% of absolute maximum (2000).
    Multi environment: use 30th and 70th percentiles of pooled distribution.
    This is because single-env agents reliably reach the maximum,
    while multi-env agents cannot — requiring empirical calibration.
    """
    if env_type == "single":
        # Original thesis methodology: fixed percentage of absolute max
        abs_max = 2000.0
        flow_low  = abs_max * 0.30
        flow_high = abs_max * 0.60
        method = "30-60% of absolute maximum (2000) — original thesis Table 3.3"
    else:
        # Empirical calibration: percentiles of pooled distribution
        pooled = np.array(all_rewards)
        flow_low  = float(np.percentile(pooled, 30))
        flow_high = float(np.percentile(pooled, 70))
        method = "30th-70th percentile of pooled static distribution (n=300)"

    return flow_low, flow_high, method


# ── Main ──────────────────────────────────────────────────────────────────── #

print("=" * 70)
print("FLOW ZONE CALIBRATION SCRIPT")
print("=" * 70)

results = {}   # env_type → list of all rewards across algos

for env_type in ENVS:
    print(f"\n{'─'*70}")
    print(f"ENVIRONMENT: {env_type.upper()}")
    print(f"{'─'*70}")

    all_env_rewards = []

    for algo, (model_cls, single_folder, multi_folder) in ALGO_CONFIG.items():
        folder = single_folder if env_type == "single" else multi_folder
        model_path = os.path.join(BASE_LOG_DIR, folder, "best_model.zip")

        if not os.path.exists(model_path):
            print(f"  ⚠️  Model not found: {model_path} — skipping")
            continue

        print(f"\n  Loading {algo.upper()} from {model_path}")
        model = model_cls.load(model_path)

        rewards = evaluate_static(
            model, env_type, N_EPISODES, FIXED_SPEED, algo
        )

        mean_r = np.mean(rewards)
        std_r  = np.std(rewards)
        min_r  = np.min(rewards)
        max_r  = np.max(rewards)
        p30    = np.percentile(rewards, 30)
        p70    = np.percentile(rewards, 70)

        print(f"\n  {algo.upper()} Static Summary ({N_EPISODES} eps, speed={FIXED_SPEED}):")
        print(f"    Mean   : {mean_r:.1f} ± {std_r:.1f}")
        print(f"    Range  : [{min_r:.1f}, {max_r:.1f}]")
        print(f"    P30/P70: [{p30:.1f}, {p70:.1f}]")

        # Save per-algo rewards
        save_path = os.path.join(
            CALIB_DIR, f"static_{env_type}_{algo}_{N_EPISODES}eps.npy"
        )
        np.save(save_path, np.array(rewards))

        all_env_rewards.extend(rewards)

    results[env_type] = all_env_rewards

# ── Compute and report bounds ─────────────────────────────────────────────── #

print(f"\n{'='*70}")
print("CALIBRATED FLOW ZONE BOUNDS")
print(f"{'='*70}")

flow_zones = {}

for env_type in ENVS:
    if env_type not in results or len(results[env_type]) == 0:
        print(f"\n{env_type.upper()}: No data available — skipping")
        continue

    flow_low, flow_high, method = compute_bounds(results[env_type], env_type)

    # Round to nearest 10 for clean presentation
    flow_low_rounded  = round(flow_low  / 10) * 10
    flow_high_rounded = round(flow_high / 10) * 10

    flow_zones[env_type] = (flow_low_rounded, flow_high_rounded)

    pooled = np.array(results[env_type])
    print(f"\n{env_type.upper()} Environment:")
    print(f"  Method    : {method}")
    print(f"  Pooled n  : {len(pooled)} episodes")
    print(f"  Pooled mean ± std: {np.mean(pooled):.1f} ± {np.std(pooled):.1f}")
    print(f"  Raw bounds      : [{flow_low:.1f}, {flow_high:.1f}]")
    print(f"  Rounded bounds  : [{flow_low_rounded}, {flow_high_rounded}]")

    # Save pooled distribution
    np.save(
        os.path.join(CALIB_DIR, f"pooled_static_{env_type}.npy"),
        pooled
    )

# ── Print ready-to-paste FLOW_ZONES dict ─────────────────────────────────── #

print(f"\n{'='*70}")
print("PASTE THIS INTO train_multi_env.py  (replace existing FLOW_ZONES dict)")
print(f"{'='*70}")
print()
print("FLOW_ZONES = {")
for env_type, (fl, fh) in flow_zones.items():
    print(f'    "{env_type}": {{')
    print(f'        "default": ({float(fl)}, {float(fh)})')
    print(f'        # Method: {"30-60% of max reward (2000)" if env_type == "single" else "30th-70th percentile of pooled static distribution (n=300)"}')
    print(f'    }},')
print("}")
print()
print(f"{'='*70}")
print("Calibration complete. Data saved to:", CALIB_DIR)
print(f"{'='*70}")