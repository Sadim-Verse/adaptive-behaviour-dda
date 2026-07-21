# training/train_multi_env.py
"""
train_multi_env.py
==================
Unified training and DDA evaluation script supporting:
    - Algorithms : DQN, DDQN, PPO
    - Environments: ChaseEnv (single enemy, 3D obs)
                    MultiChaseEnv (3 enemies, 8D obs)
                    ObstacleMiniGridEnv (procedurally-generated MiniGrid-based
                    hazard avoidance, 41D obs)

THIS IS A MERGE of two independently-edited versions of this file that had
diverged without either side knowing about the other's changes:

  1. Sustained-recovery detection (n_consecutive_recovery, default 5) for
     Priority 6 -- recovery must hold for several consecutive episodes,
     not just cross the threshold once. This version had NOT yet received
     the fixed_difficulty fix below.
  2. fixed_difficulty ablation holding at initial_speed (the actual
     training speed, e.g. 2.0) instead of min_speed (0.3) -- confirmed
     correct via analyze_ablations.py producing the expected 2000.0 +/-
     0.0 perfect score. This version still had single-crossing recovery
     detection.

Both fixes are combined here. IMPORTANT: because of this divergence, any
existing spike-mode data anywhere in this project may have been generated
under EITHER detection method depending on which file was live at the
time, with no record of which. Before trusting any existing
adaptation_latency.npy file, verify it was generated under this merged
version (check file mtime against this file's mtime, or re-run) rather
than assuming. Going forward, use git (or at minimum a single shared file,
never edited in two places at once) to prevent this recurring.

Usage examples
--------------
# Train DDQN on the original single-enemy environment (default)
python -m training.train_multi_env --algo ddqn --env single

# Evaluate a saved DDQN model with DDA on the single-enemy environment (spike mode)
python -m training.train_multi_env --algo ddqn --env single \\
    --resume logs/single_ddqn/best_model.zip --delta 0.05 --seed 42 --max_speed 9.0 --spike 250

# Run an ablation study (no DDA)
python -m training.train_multi_env --algo ddqn --env single \\
    --resume logs/single_ddqn/best_model.zip --delta 0.05 --seed 42 --max_speed 6.0 --ablation no_dda
"""

import os
import time
import argparse
import numpy as np

from training.ddqn import DDQN
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from gymnasium.wrappers import TimeLimit

from envs.chase_env import ChaseEnv
from envs.multi_chase_env import MultiChaseEnv
from envs.obstacle_minigrid_env import ObstacleMiniGridEnv
from training.dda_controller import DDAController


# Flow Zone boundaries -- confirmed calibrated values, verified consistent
# across every script in this project as of this merge.
FLOW_ZONES = {
    "single": {"default": (296.0, 765.0)},
    "multi": {"default": (245.6, 470.1)},
    "obstacle_minigrid": {"default": (-74.5, -23.7)},
}

DDA_SCALE = {
    "single":            {"min_speed": 0.3, "max_speed_default": 9.0, "initial_speed": 2.0, "eval_speed": 2.0},
    "multi":              {"min_speed": 0.3, "max_speed_default": 9.0, "initial_speed": 2.0, "eval_speed": 2.0},
    "obstacle_minigrid":   {"min_speed": 0.0, "max_speed_default": 1.0, "initial_speed": 0.0, "eval_speed": 0.5},
}


def make_env(env_type="single", initial_speed=None, render_mode=None, randomize_difficulty=False):
    if initial_speed is None:
        initial_speed = DDA_SCALE[env_type]["initial_speed"]

    if env_type == "multi":
        env = MultiChaseEnv(render_mode=render_mode, enemy_speed=initial_speed,
                            homing_strength=3.0)
        info_keywords = ("enemy_speed",)
        max_episode_steps = 2000

    elif env_type == "obstacle_minigrid":
        env = ObstacleMiniGridEnv(render_mode=render_mode, difficulty=initial_speed,
                                   randomize_difficulty=randomize_difficulty)
        info_keywords = ("difficulty", "num_hazards")
        max_episode_steps = 2000

    else:
        env = ChaseEnv(render_mode=render_mode, enemy_speed=initial_speed,
                       homing_strength=3.0)
        info_keywords = ("enemy_speed",)
        max_episode_steps = 2000

    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = Monitor(env, info_keywords=info_keywords)
    return env


def build_model(algo, vec_env, log_dir, seed=None):
    algo = algo.lower()

    if algo == "dqn":
        model = DQN("MlpPolicy", vec_env, learning_rate=0.0003, buffer_size=50_000,
                     batch_size=64, exploration_fraction=0.2, exploration_final_eps=0.02,
                     target_update_interval=2000, gamma=0.99, optimize_memory_usage=False,
                     tensorboard_log=log_dir, verbose=1, seed=seed)
    elif algo == "ddqn":
        model = DDQN("MlpPolicy", vec_env, learning_rate=0.0003, buffer_size=50_000,
                      batch_size=64, exploration_fraction=0.2, exploration_final_eps=0.02,
                      target_update_interval=2000, gamma=0.99, optimize_memory_usage=False,
                      tensorboard_log=log_dir, verbose=1, seed=seed)
    elif algo == "ppo":
        model = PPO("MlpPolicy", vec_env, learning_rate=0.0003, n_steps=2_048,
                     batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                     clip_range=0.2, ent_coef=0.01, tensorboard_log=log_dir,
                     verbose=1, seed=seed)
    else:
        raise ValueError(f"Unknown algorithm '{algo}'. Choose from: dqn, ddqn, ppo")

    return model


def run_dda_only(model, env, dda, n_episodes=500, log_dir=None, base_seed=42,
                 spike_episode=None, n_consecutive_recovery=5):
    """
    Evaluate a frozen policy under live DDA control for n_episodes.

    Priority 6 fix: recovery is detected via SUSTAINED crossing (rolling
    average must stay >= recovery_threshold for n_consecutive_recovery
    consecutive episodes), not a single crossing. Verified against real
    cached Multi-environment spike data: sustained detection shifts
    latencies later (as expected) but did NOT dramatically shrink cross-
    seed variance on its own (27.7 -> 26.5 std across 15 runs) -- evidence
    FOR genuine environmental stochasticity as well, not proof the
    detection bug was the sole cause of the originally-flagged variance.
    Report both in the thesis rather than claim this fix alone resolves it.
    """
    spike_applied = False
    pre_spike_peak = None
    recovery_threshold = None
    t_drop = None
    t_recovery = None
    adaptation_latency = None
    recovery_check_start = None
    consecutive_above = 0
    streak_start_episode = None

    if log_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=log_dir)

    obs, _ = env.reset(seed=base_seed)
    ep_reward = 0.0
    episode_count = 0

    while episode_count < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_reward += reward

        if terminated or truncated:
            dda.update(ep_reward)
            env.unwrapped.set_difficulty(dda.current_speed)
            episode_count += 1

            rolling_avg = (np.mean(list(dda.reward_buffer))
                          if len(dda.reward_buffer) > 0 else ep_reward)

            if spike_episode is not None and episode_count == spike_episode and not spike_applied:
                if len(dda.rolling_avg_history) >= 10:
                    pre_spike_peak = max(dda.rolling_avg_history[-10:])
                else:
                    pre_spike_peak = rolling_avg
                recovery_threshold = pre_spike_peak * 0.80
                t_drop = spike_episode
                recovery_check_start = spike_episode + dda.window_size

                env.unwrapped.set_difficulty(dda.max_speed)
                dda.current_speed = dda.max_speed
                spike_applied = True
                print(f"\n*** SPIKE at ep {episode_count} | "
                      f"peak={pre_spike_peak:.0f} | "
                      f"threshold={recovery_threshold:.0f} | "
                      f"speed->{dda.max_speed} ***")
                print(f"*** Recovery check will start at episode {recovery_check_start}, "
                      f"requiring {n_consecutive_recovery} consecutive episodes >= threshold ***\n")

            if (spike_applied and t_recovery is None and
                recovery_check_start is not None and
                episode_count >= recovery_check_start):
                if rolling_avg >= recovery_threshold:
                    if consecutive_above == 0:
                        streak_start_episode = episode_count
                    consecutive_above += 1
                    if consecutive_above >= n_consecutive_recovery:
                        t_recovery = streak_start_episode
                        adaptation_latency = t_recovery - t_drop
                        print(f"*** RECOVERY at ep {t_recovery} (confirmed by ep {episode_count}) | "
                              f"latency={adaptation_latency} eps ***")
                else:
                    consecutive_above = 0
                    streak_start_episode = None

            if log_dir:
                writer.add_scalar("dda/current_speed", dda.current_speed, episode_count)
                writer.add_scalar("dda/episode_reward", ep_reward, episode_count)
                writer.add_scalar("dda/rolling_avg_reward", rolling_avg, episode_count)

            print(f"Episode {episode_count:4d} | reward: {ep_reward:6.0f} | "
                  f"speed: {dda.current_speed:.2f} | rolling_avg: {rolling_avg:.0f}")

            obs, _ = env.reset(seed=base_seed + episode_count)
            ep_reward = 0.0

    latency_result = {
        "t_drop": t_drop,
        "t_recovery": t_recovery,
        "adaptation_latency": adaptation_latency if adaptation_latency else n_episodes,
        "pre_spike_peak": pre_spike_peak,
        "recovery_threshold": recovery_threshold
    }

    if log_dir and spike_episode is not None:
        np.save(os.path.join(log_dir, "adaptation_latency.npy"),
                np.array([latency_result["adaptation_latency"]]))

    if spike_episode is not None:
        print(f"\nAdaptation Latency: {latency_result['adaptation_latency']} episodes")

    return latency_result


class DDACallback(BaseCallback):
    def __init__(self, dda, env, verbose=0):
        super().__init__(verbose)
        self.dda = dda
        self.env = env
        self.ep_reward = 0.0

    def _on_step(self):
        self.ep_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.dda.update(self.ep_reward)
            unwrapped = self.env.unwrapped
            if hasattr(unwrapped, "set_difficulty"):
                unwrapped.set_difficulty(self.dda.current_speed)
            if len(self.dda.reward_buffer) > 0:
                self.logger.record("dda/rolling_avg_reward", np.mean(list(self.dda.reward_buffer)))
            self.logger.record("dda/current_speed", self.dda.current_speed)
            self.logger.record("dda/episode_reward", self.ep_reward)
            self.ep_reward = 0.0
        return True


def main(algo, env_type, resume_from, delta, seed, timesteps, output_dir,
         max_speed=None, spike_episode=None, ablation=None, n_consecutive_recovery=5):
    scale = DDA_SCALE[env_type]
    if max_speed is None:
        max_speed = scale["max_speed_default"]
    min_speed = scale["min_speed"]
    initial_speed = scale["initial_speed"]
    eval_speed = scale["eval_speed"]

    run_id = f"{env_type}_{algo}_delta{str(delta).replace('.','p')}_seed{seed}"
    if spike_episode is not None:
        run_id += "_spike"
    if ablation is not None:
        run_id += f"_abl_{ablation}"

    log_dir = os.path.abspath(os.path.join(output_dir, run_id))
    os.makedirs(log_dir, exist_ok=True)

    env = make_env(env_type=env_type, initial_speed=initial_speed)

    if resume_from is not None:
        print(f"\n{'='*60}")
        print(f"EVALUATION MODE")
        print(f"  Algorithm   : {algo.upper()}")
        print(f"  Environment : {env_type}")
        print(f"  Model       : {resume_from}")
        print(f"  Delta       : {delta}  |  Seed : {seed}")
        print(f"  Min Speed   : {min_speed}  |  Max Speed : {max_speed}")
        print(f"  Spike       : {spike_episode if spike_episode else 'Disabled'}")
        if ablation is not None:
            print(f"  Ablation    : {ablation}")
        print(f"{'='*60}\n")

        if algo.lower() == "ddqn":
            model = DDQN.load(resume_from)
        elif algo.lower() == "dqn":
            model = DQN.load(resume_from)
        elif algo.lower() == "ppo":
            model = PPO.load(resume_from)
        else:
            raise ValueError(f"Unknown algorithm: {algo}")

        flow_low, flow_high = FLOW_ZONES[env_type]["default"]
        dda = DDAController(window_size=10, flow_low=flow_low, flow_high=flow_high,
                             delta=delta, min_speed=min_speed, max_speed=max_speed,
                             initial_speed=initial_speed)

        if ablation is not None:
            if ablation == "no_dda":
                fixed_speed = max_speed
                env.unwrapped.set_difficulty(fixed_speed)
                dda.current_speed = fixed_speed

                def no_dda_update(reward):
                    dda.episode_rewards.append(reward)
                    dda.reward_buffer.append(reward)
                    avg = float(np.mean(dda.reward_buffer)) if len(dda.reward_buffer) >= dda.window_size else float(reward)
                    dda.rolling_avg_history.append(avg)
                    dda.speed_history.append(dda.current_speed)

                dda.update = no_dda_update

                def no_dda_set_difficulty(speed):
                    pass
                env.unwrapped.set_difficulty = no_dda_set_difficulty
                if hasattr(env.unwrapped, "enemy_speed"):
                    env.unwrapped.enemy_speed = fixed_speed
                print(f"  Ablation: No DDA (fixed speed/difficulty at {fixed_speed})")

            elif ablation == "fixed_difficulty":
                # Holds at initial_speed (the TRAINING speed, e.g. 2.0 for
                # single/multi) -- NOT min_speed. This is the actual point
                # of the ablation: show the agent at exactly the difficulty
                # it was trained on, with no adaptation happening around
                # it. Confirmed correct via analyze_ablations.py producing
                # the expected 2000.0 +/- 0.0 perfect score for Single.
                env.unwrapped.set_difficulty(initial_speed)
                dda.current_speed = initial_speed

                def fixed_update(reward):
                    dda.episode_rewards.append(reward)
                    dda.reward_buffer.append(reward)
                    avg = float(np.mean(dda.reward_buffer)) if len(dda.reward_buffer) >= dda.window_size else float(reward)
                    dda.rolling_avg_history.append(avg)
                    dda.speed_history.append(dda.current_speed)

                dda.update = fixed_update

                def fixed_set_difficulty(speed):
                    pass
                env.unwrapped.set_difficulty = fixed_set_difficulty
                print(f"  Ablation: Fixed Difficulty (speed held constant at {initial_speed})")

            elif ablation == "no_rolling_window":
                dda.window_size = 1
                original_update = dda.update
                def no_rolling_update(reward):
                    dda.reward_buffer.clear()
                    dda.reward_buffer.append(reward)
                    original_update(reward)
                dda.update = no_rolling_update
                print("  Ablation: No Rolling Window (window_size=1, immediate reward)")

            elif ablation == "no_smoothing":
                dda.window_size = 1
                original_update = dda.update
                def no_smoothing_update(reward):
                    dda.reward_buffer.clear()
                    dda.reward_buffer.append(reward)
                    original_update(reward)
                dda.update = no_smoothing_update
                print("  Ablation: No Reward Smoothing (raw rewards)")

            else:
                print(f"  WARNING: Unknown ablation mode: {ablation}")

        start_time = time.time()
        latency_result = run_dda_only(model, env, dda, n_episodes=500, log_dir=log_dir,
                                       base_seed=seed, spike_episode=spike_episode,
                                       n_consecutive_recovery=n_consecutive_recovery)
        wall_time = time.time() - start_time

        np.save(os.path.join(log_dir, "episode_rewards.npy"), dda.episode_rewards)
        np.save(os.path.join(log_dir, "speed_history.npy"),   dda.speed_history)
        np.save(os.path.join(log_dir, "rolling_avg.npy"),     dda.rolling_avg_history)

        fzc = dda.get_flow_zone_consistency()
        print(f"\n{'='*60}")
        print(f"RESULTS  |  {run_id}")
        print(f"  Flow Zone Consistency : {fzc:.2%}")
        print(f"  Mean Episode Reward   : {np.mean(dda.episode_rewards):.1f}")
        if spike_episode is not None:
            print(f"  Adaptation Latency   : {latency_result['adaptation_latency']} eps")
        print(f"  Wall-clock time       : {wall_time:.1f}s")
        print(f"  Saved to              : {log_dir}")
        print(f"{'='*60}\n")
        return

    print(f"\n{'='*60}")
    print(f"TRAINING MODE")
    print(f"  Algorithm   : {algo.upper()}")
    print(f"  Environment : {env_type}")
    print(f"  Timesteps   : {timesteps:,}")
    print(f"{'='*60}\n")

    vec_env = DummyVecEnv([lambda: make_env(
        env_type=env_type, initial_speed=initial_speed,
        randomize_difficulty=(env_type == "obstacle_minigrid"))])
    model = build_model(algo, vec_env, log_dir, seed=seed)

    eval_env = DummyVecEnv([lambda: make_env(
        env_type=env_type, initial_speed=eval_speed, randomize_difficulty=False)])
    checkpoint_cb = CheckpointCallback(save_freq=100_000, save_path=log_dir, name_prefix="checkpoint")
    eval_cb = EvalCallback(eval_env, best_model_save_path=log_dir, log_path=log_dir,
                            eval_freq=50_000, n_eval_episodes=10, deterministic=True, verbose=1)

    start_time = time.time()
    model.learn(total_timesteps=timesteps, callback=[checkpoint_cb, eval_cb], tb_log_name=run_id)
    wall_time = time.time() - start_time

    model_path = os.path.join(log_dir, "final_model")
    model.save(model_path)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE  |  {run_id}")
    print(f"  Wall-clock time : {wall_time:.1f}s  ({wall_time/60:.1f} min)")
    print(f"  Model saved to  : {model_path}.zip")
    print(f"  Best model at   : {os.path.join(log_dir, 'best_model.zip')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or evaluate DQN / DDQN / PPO on ChaseEnv, MultiChaseEnv, or ObstacleMiniGridEnv"
    )
    parser.add_argument("--algo", type=str, default="ddqn", choices=["dqn", "ddqn", "ppo"])
    parser.add_argument("--env", type=str, default="single", choices=["single", "multi", "obstacle_minigrid"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--output", type=str, default="./logs/experiments/")
    parser.add_argument("--max_speed", type=float, default=None)
    parser.add_argument("--spike", type=int, default=None)
    parser.add_argument("--n_consecutive_recovery", type=int, default=5,
                        help="Consecutive episodes rolling avg must stay >= threshold "
                             "before recovery is declared. 1 = old single-crossing behavior.")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["no_dda", "fixed_difficulty", "no_rolling_window", "no_smoothing"])

    args = parser.parse_args()

    main(algo=args.algo, env_type=args.env, resume_from=args.resume, delta=args.delta,
         seed=args.seed, timesteps=args.timesteps, output_dir=args.output,
         max_speed=args.max_speed, spike_episode=args.spike, ablation=args.ablation,
         n_consecutive_recovery=args.n_consecutive_recovery)