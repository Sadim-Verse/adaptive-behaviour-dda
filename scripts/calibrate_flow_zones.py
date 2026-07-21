"""
scripts/calibrate_flow_zones.py
================================
Empirically calibrates Flow Zone bounds for each environment by running
STATIC (no-DDA, fixed-difficulty) baseline evaluations of each trained,
frozen model, then applying ONE consistent percentile rule to the pooled
reward distribution.

Why static, not DDA-active
--------------------------
Calibrating bounds from DDA-ACTIVE evaluation runs is circular: those
episodes are produced by a controller already steering difficulty toward
whatever the CURRENT bounds are, so deriving new bounds from that data is
self-referential. Static baseline runs (frozen policy, fixed difficulty,
no controller involved) measure the agent's inherent capability at a
neutral reference point, independent of any adaptive mechanism -- which is
what "the range within which the agent is optimally challenged" actually
requires to be a meaningful, non-arbitrary statement.

Why one consistent rule, not per-environment special cases
------------------------------------------------------------
The previous version of this script used TWO different methods: Single
got a fixed 30-60% of its theoretical max (2000), justified in code as
"single-env agents reliably reach the maximum" -- true for the ORIGINAL
weak environment, but not a real calibration, just a shortcut that seemed
safe because those agents were already near-perfect. Multi got an actual
empirical 30th-70th percentile. This inconsistency is exactly what
Priority 3 (formalize Flow Zone mathematically) asks to fix. This version
applies ONE rule -- 40th-70th percentile of the pooled static
distribution -- to every environment. The resulting ABSOLUTE bounds will
still differ per environment (that's expected: each environment's reward
scale is different), but the RULE producing them is now identical, and by
construction each environment's zone captures exactly 30% of its own
pooled distribution -- a fair, comparable baseline coverage regardless of
scale. (40-70, not 40-80, specifically because 40-80 was found to
saturate at Single's reward ceiling: ~29% of pooled Single episodes hit
the exact max reward, so any percentile >=~72 collapses to that same
degenerate value.)

Usage:
    python -m scripts.calibrate_flow_zones
    python -m scripts.calibrate_flow_zones --envs single multi obstacle_minigrid

Output:
    - Prints the calibrated bounds for each environment
    - Saves pooled calibration data to logs/calibration/
    - Prints the exact FLOW_ZONES dict to paste into train_multi_env.py
"""

import os
import argparse
import numpy as np
from stable_baselines3 import DQN, PPO
from training.ddqn import DDQN
from training.train_multi_env import make_env, DDA_SCALE

# ── Configuration ─────────────────────────────────────────────────────────── #

N_EPISODES   = 100      # episodes per algorithm per environment
# Different environments' runs live under different folders in this
# project (single/multi under the original convention; obstacle_minigrid
# was trained with --output ./logs/procedural_env_v2/). A single global
# BASE_LOG_DIR would silently produce "Model not found" for whichever
# environment doesn't match it -- mapped per-environment instead.
BASE_LOG_DIRS = {
    "single": "./logs/experiments",
    "multi": "./logs/experiments",
    "obstacle_minigrid": "./logs/procedural_env_v2",
}
CALIB_DIR    = "./logs/calibration"
os.makedirs(CALIB_DIR, exist_ok=True)

MODEL_CLASSES = {"dqn": DQN, "ddqn": DDQN, "ppo": PPO}
ALGOS = ["dqn", "ddqn", "ppo"]

# The formal, consistent percentile rule -- see module docstring for why
# 40-70 rather than 40-80 or 30-70.
PERCENTILE_LOW  = 40
PERCENTILE_HIGH = 70

# Calibration needs a difficulty point that actually produces VARIANCE in
# the reward distribution -- a point where the agent sometimes succeeds
# and sometimes doesn't, not one it trivially solves every time. This is
# DELIBERATELY separate from DDA_SCALE's "eval_speed" (used by
# EvalCallback during training to track progress at one fixed point, a
# different purpose). Confirmed empirically: Single's agents get a
# perfect, zero-variance 2000.0 on every single episode at eval_speed=2.0
# -- calibrating a percentile-based zone from a distribution with zero
# variance is mathematically degenerate (collapses to a single point).
# At ChaseEnv's actual maximum difficulty (6.0, the ceiling set_difficulty()
# clips to), real variance appears (DQN: 495.9+/-495.7, DDQN: 972.5+/-755.0,
# PPO: 379.1+/-307.3) -- meaning 6.0, not 2.0, is where Single's
# calibration is actually meaningful.
CALIBRATION_SPEED_OVERRIDE = {
    "single": 6.0,
    # multi and obstacle_minigrid already show real variance at their
    # DDA_SCALE eval_speed, no override needed.
}


def model_path_for(env_type, algo):
    """
    Matches the ACTUAL folder layout in this project, which differs
    between environments:
      single/multi:        logs/experiments/{algo}/{env}_{algo}_delta0p05_seed42/best_model.zip
      obstacle_minigrid:    logs/procedural_env_v2/{env}_{algo}_delta0p05_seed42/best_model.zip
    (single/multi have an extra {algo}-named subfolder level that
    obstacle_minigrid does not -- a real, existing inconsistency in how
    the runs were organized, not something to silently paper over.)
    """
    folder = f"{env_type}_{algo}_delta0p05_seed42"
    base = BASE_LOG_DIRS[env_type]
    if env_type in ("single", "multi"):
        return os.path.join(base, algo, folder, "best_model.zip")
    return os.path.join(base, folder, "best_model.zip")


def evaluate_static(model, env_type, n_episodes, fixed_speed, algo_name):
    """
    Run frozen policy for n_episodes at a FIXED difficulty, no DDA
    Controller involved. This is the "neutral reference point" data
    source -- see module docstring for why this (not DDA-active data)
    is the legitimate calibration source.
    """
    # randomize_difficulty=False is the make_env() default for
    # single/multi already; explicit here for obstacle_minigrid so
    # calibration measures capability at ONE fixed representative point,
    # not smeared across the training-time randomized distribution.
    env = make_env(env_type=env_type, initial_speed=fixed_speed,
                   randomize_difficulty=False)
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
    ONE consistent rule for every environment: 40th and 70th percentile
    of the pooled static-baseline reward distribution.

    No per-environment special casing. Absolute numbers will still differ
    across environments (expected -- reward scales differ), but the RULE
    is identical, and each environment's zone captures exactly the same
    30% baseline coverage of its own distribution by construction.
    """
    pooled = np.asarray(all_rewards)
    flow_low  = float(np.percentile(pooled, PERCENTILE_LOW))
    flow_high = float(np.percentile(pooled, PERCENTILE_HIGH))
    method = f"{PERCENTILE_LOW}th-{PERCENTILE_HIGH}th percentile of pooled static baseline (n={len(pooled)})"

    # Sanity check against the ceiling-saturation problem found for
    # Single -- warn (don't silently produce a degenerate bound) if this
    # ever recurs for any environment.
    ceiling = pooled.max()
    frac_at_ceiling = float(np.mean(pooled >= ceiling - 1e-6))
    if flow_high >= ceiling - 1e-6:
        print(f"  ⚠️  WARNING: upper bound ({flow_high:.1f}) is saturated at the "
              f"observed max ({ceiling:.1f}). {frac_at_ceiling*100:.1f}% of episodes "
              f"hit this value. This percentile choice is degenerate for this "
              f"environment -- consider a lower PERCENTILE_HIGH.")

    return flow_low, flow_high, method


# ── Main ──────────────────────────────────────────────────────────────────── #

def main(envs):
    print("=" * 70)
    print("FLOW ZONE CALIBRATION SCRIPT")
    print(f"Rule: {PERCENTILE_LOW}th-{PERCENTILE_HIGH}th percentile, applied identically to all environments")
    print("=" * 70)

    results = {}   # env_type → list of all rewards across algos

    for env_type in envs:
        print(f"\n{'─'*70}")
        print(f"ENVIRONMENT: {env_type.upper()}")
        print(f"{'─'*70}")

        fixed_speed = CALIBRATION_SPEED_OVERRIDE.get(env_type, DDA_SCALE[env_type]["eval_speed"])
        print(f"  Static baseline difficulty (fixed, no DDA): {fixed_speed}")

        all_env_rewards = []

        for algo in ALGOS:
            model_cls = MODEL_CLASSES[algo]
            model_path = model_path_for(env_type, algo)

            if not os.path.exists(model_path):
                print(f"  ⚠️  Model not found: {model_path} — skipping")
                continue

            print(f"\n  Loading {algo.upper()} from {model_path}")
            model = model_cls.load(model_path)

            rewards = evaluate_static(model, env_type, N_EPISODES, fixed_speed, algo)

            mean_r, std_r = np.mean(rewards), np.std(rewards)
            min_r, max_r = np.min(rewards), np.max(rewards)
            p_lo = np.percentile(rewards, PERCENTILE_LOW)
            p_hi = np.percentile(rewards, PERCENTILE_HIGH)

            print(f"\n  {algo.upper()} Static Summary ({N_EPISODES} eps, difficulty={fixed_speed}):")
            print(f"    Mean   : {mean_r:.1f} ± {std_r:.1f}")
            print(f"    Range  : [{min_r:.1f}, {max_r:.1f}]")
            print(f"    P{PERCENTILE_LOW}/P{PERCENTILE_HIGH}: [{p_lo:.1f}, {p_hi:.1f}]")

            save_path = os.path.join(CALIB_DIR, f"static_{env_type}_{algo}_{N_EPISODES}eps.npy")
            np.save(save_path, np.array(rewards))

            all_env_rewards.extend(rewards)

        results[env_type] = all_env_rewards

    # ── Compute and report bounds ─────────────────────────────────────── #

    print(f"\n{'='*70}")
    print("CALIBRATED FLOW ZONE BOUNDS")
    print(f"{'='*70}")

    flow_zones = {}

    for env_type in envs:
        if env_type not in results or len(results[env_type]) == 0:
            print(f"\n{env_type.upper()}: No data available — skipping")
            continue

        flow_low, flow_high, method = compute_bounds(results[env_type], env_type)

        pooled = np.array(results[env_type])
        print(f"\n{env_type.upper()} Environment:")
        print(f"  Method    : {method}")
        print(f"  Pooled n  : {len(pooled)} episodes")
        print(f"  Pooled mean ± std: {np.mean(pooled):.1f} ± {np.std(pooled):.1f}")
        print(f"  Bounds    : [{flow_low:.1f}, {flow_high:.1f}]")

        flow_zones[env_type] = (round(flow_low, 1), round(flow_high, 1))

        np.save(os.path.join(CALIB_DIR, f"pooled_static_{env_type}.npy"), pooled)

    # ── Print ready-to-paste FLOW_ZONES dict ──────────────────────────── #

    print(f"\n{'='*70}")
    print("PASTE THIS INTO train_multi_env.py  (replace existing FLOW_ZONES dict)")
    print(f"{'='*70}\n")
    print("FLOW_ZONES = {")
    for env_type, (fl, fh) in flow_zones.items():
        print(f'    "{env_type}": {{')
        print(f'        "default": ({fl}, {fh})')
        print(f'        # Method: {PERCENTILE_LOW}th-{PERCENTILE_HIGH}th percentile of pooled')
        print(f'        #         STATIC baseline (no DDA), n={len(results[env_type])} -- SAME rule')
        print(f'        #         applied to every environment, see calibrate_flow_zones.py')
        print(f'    }},')
    print("}")
    print(f"\n{'='*70}")
    print("Calibration complete. Data saved to:", CALIB_DIR)
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", default=["single", "multi", "obstacle_minigrid"],
                        choices=["single", "multi", "obstacle_minigrid"],
                        help="Which environments to calibrate")
    args = parser.parse_args()
    main(args.envs)