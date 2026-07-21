# scripts/extend_training.py
"""
Resume training for one or more (algorithm, environment) pairs from their
existing best_model.zip checkpoint, up to a new total timestep target.

Generalizes the original single-hardcoded-run version (DQN, multi only) to
match the current project scope: 3 algorithms x 3 environments (though in
practice you'll mainly use this for obstacle_minigrid, since none of the
curves for single/multi showed a reason to extend).

CRITICAL for obstacle_minigrid specifically: the training env MUST be
recreated with randomize_difficulty=True, exactly as in the original
training run. If you resume with a fixed-difficulty env instead, the
model will silently start overfitting to one difficulty point for the
remaining timesteps, quietly undoing the domain-randomization fix.

Usage:
    python -m scripts.extend_training --algo ppo --env obstacle_minigrid --target_timesteps 10000000
    python -m scripts.extend_training --algo dqn --env obstacle_minigrid --algo ddqn --env obstacle_minigrid --target_timesteps 10000000
    # (repeat --algo/--env pairs, or use --all to extend all three algorithms for one environment)
"""

import os
import argparse
from stable_baselines3 import DQN, PPO
from training.ddqn import DDQN
from training.train_multi_env import make_env, DDA_SCALE
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

MODEL_CLASSES = {"dqn": DQN, "ddqn": DDQN, "ppo": PPO}
BASE_LOG_DIR = "./logs/procedural_env_v2"


def extend_one(algo, env_type, target_timesteps, base_log_dir=BASE_LOG_DIR,
                checkpoint_name="final_model.zip"):
    """
    Resume one (algo, env_type) run from its saved checkpoint up to
    target_timesteps TOTAL (not additional) -- e.g. target_timesteps=10_000_000
    on a model already trained to 6,000,000 runs 4,000,000 more steps.

    IMPORTANT: defaults to final_model.zip, NOT best_model.zip.
    best_model.zip is overwritten by EvalCallback every time a new best
    EVALUATION score is found -- it's a snapshot from WHENEVER that peak
    occurred during training, not from the end of the run. Resuming from
    it silently discards everything learned after that peak and makes
    the model.num_timesteps count (and therefore the "additional steps"
    calculation below) reflect that earlier point, not the true training
    duration. final_model.zip is saved exactly once, at the actual end of
    training, and is what "already trained" should mean when the goal is
    to CONTINUE training rather than deploy the best-scoring snapshot.

    NOTE: resuming does NOT restore the replay buffer (DQN/DDQN only --
    PPO has no replay buffer). SB3's model.save() doesn't persist it by
    default. Expect a brief, normal transient dip in performance right
    after resuming while the buffer refills -- this is a known, standard
    limitation of checkpoint resumption, not a new bug.
    """
    run_id = f"{env_type}_{algo}_delta0p05_seed42"
    log_dir = os.path.join(base_log_dir, run_id)
    model_path = os.path.join(log_dir, checkpoint_name)

    if not os.path.exists(model_path):
        print(f"⚠️  Skipping {algo}/{env_type}: checkpoint not found at {model_path}")
        if checkpoint_name == "final_model.zip":
            print(f"    (if only best_model.zip exists, the original training run may not "
                  f"have completed / saved a final model -- check the original training log)")
        return

    model_cls = MODEL_CLASSES[algo.lower()]
    initial_speed = DDA_SCALE[env_type]["initial_speed"]

    # CRITICAL: randomize_difficulty must match how the run was ORIGINALLY
    # trained. For obstacle_minigrid that's True (domain randomization);
    # for single/multi it's a no-op (False), matching their existing,
    # unchanged training methodology.
    randomize = (env_type == "obstacle_minigrid")

    train_env = DummyVecEnv([lambda: make_env(
        env_type=env_type, initial_speed=initial_speed, randomize_difficulty=randomize
    )])

    print(f"\n{'='*60}")
    print(f"EXTENDING: {algo.upper()} on {env_type}")
    print(f"  Checkpoint       : {model_path}")
    print(f"  randomize_difficulty : {randomize}")
    print(f"  Target timesteps : {target_timesteps:,}")
    print(f"{'='*60}\n")

    model = model_cls.load(model_path, env=train_env)
    already_trained = model.num_timesteps
    additional = target_timesteps - already_trained

    if additional <= 0:
        print(f"  Already at or past {target_timesteps:,} steps "
              f"(currently {already_trained:,}) -- nothing to do.")
        return

    print(f"  Already trained  : {already_trained:,} steps")
    print(f"  Additional steps : {additional:,}")

    # eval_env, same fixed representative-difficulty convention as the
    # original training run (NOT randomized -- EvalCallback needs a
    # stable difficulty for checkpoint-to-checkpoint comparability).
    eval_speed = DDA_SCALE[env_type]["eval_speed"]
    eval_env = DummyVecEnv([lambda: make_env(
        env_type=env_type, initial_speed=eval_speed, randomize_difficulty=False
    )])

    checkpoint_cb = CheckpointCallback(
        save_freq=100_000, save_path=log_dir, name_prefix="checkpoint_extended"
    )
    eval_cb = EvalCallback(
        eval_env, best_model_save_path=log_dir, log_path=log_dir,
        eval_freq=50_000, n_eval_episodes=10, deterministic=True, verbose=1
    )

    model.learn(
        total_timesteps=additional,
        reset_num_timesteps=False,   # keeps the TensorBoard timeline continuous
        callback=[checkpoint_cb, eval_cb],
        tb_log_name=run_id,
    )

    final_path = os.path.join(log_dir, "final_model_extended.zip")
    model.save(final_path)
    print(f"\n✅ Extended {algo.upper()}/{env_type} to {model.num_timesteps:,} steps.")
    print(f"   Final model : {final_path}")
    print(f"   Best model  : {os.path.join(log_dir, 'best_model.zip')} "
          f"(overwritten if a later checkpoint improved on it)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", action="append", required=True,
                        choices=["dqn", "ddqn", "ppo"],
                        help="Repeat --algo for multiple algorithms")
    parser.add_argument("--env", type=str, required=True,
                        choices=["single", "multi", "obstacle_minigrid"])
    parser.add_argument("--target_timesteps", type=int, default=10_000_000,
                        help="TOTAL timesteps to reach (not additional)")
    parser.add_argument("--output", type=str, default=BASE_LOG_DIR)
    parser.add_argument("--checkpoint", type=str, default="final_model.zip",
                        help="Which checkpoint file to resume from (default: final_model.zip, "
                             "the true end-of-run state). Only use best_model.zip if you "
                             "specifically want to continue from the best-scoring snapshot "
                             "instead of the actual training endpoint -- see extend_one()'s "
                             "docstring for why these are NOT interchangeable.")
    args = parser.parse_args()

    for algo in args.algo:
        extend_one(algo, args.env, args.target_timesteps, base_log_dir=args.output,
                   checkpoint_name=args.checkpoint)


if __name__ == "__main__":
    main()