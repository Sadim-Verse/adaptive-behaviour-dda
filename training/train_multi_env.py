# training/train_multi_env.py
"""
train_multi_env.py
==================
Unified training and DDA evaluation script supporting:
    - Algorithms : DQN, DDQN, PPO
    - Environments: ChaseEnv (single enemy, 3D obs)
                    MultiChaseEnv (3 enemies, 8D obs)

Usage examples
--------------
# Train DDQN on the original single-enemy environment (default)
python -m training.train_multi_env --algo ddqn --env single

# Train PPO on the multi-enemy environment
python -m training.train_multi_env --algo ppo --env multi

# Evaluate a saved DDQN model with DDA on the single-enemy environment (steady mode)
python -m training.train_multi_env --algo ddqn --env single \\
    --resume logs/single_ddqn/best_model.zip --delta 0.05 --seed 42 --max_speed 6.0

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

from training.ddqn import DDQN  # For DDQN
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from gymnasium.wrappers import TimeLimit

from envs.chase_env import ChaseEnv
from envs.multi_chase_env import MultiChaseEnv
from training.dda_controller import DDAController


# Flow Zone boundaries
FLOW_ZONES = {
    "single": {
        "default": (600.0, 1200.0)   # Original thesis: 30-60% of 2000 max
    },
    "multi": {
        "default": (220.0, 470.0)    # Empirical 30th-70th percentile (pooled 300 episodes)
    }
}


# ═══════════════════════════════════════════════════════════════════════════ #
#  Environment factory                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

def make_env(env_type="single", initial_speed=2.0, render_mode=None):
    """
    Instantiate and wrap the chosen environment.

    Parameters
    ----------
    env_type : str
        "single" → ChaseEnv (3D obs, 1 enemy)
        "multi"  → MultiChaseEnv (8D obs, 3 enemies)
    initial_speed : float
        Starting enemy speed passed to the environment constructor.
    render_mode : str or None
        Pass "human" to enable graphical rendering.
    """
    if env_type == "multi":
        env = MultiChaseEnv(render_mode=render_mode, enemy_speed=initial_speed,
                            homing_strength=3.0)
    else:
        env = ChaseEnv(render_mode=render_mode, enemy_speed=initial_speed,
                       homing_strength=3.0)

    env = TimeLimit(env, max_episode_steps=2000)
    env = Monitor(env, info_keywords=("enemy_speed",))
    return env


# ═══════════════════════════════════════════════════════════════════════════ #
#  Algorithm factory                                                          #
# ═══════════════════════════════════════════════════════════════════════════ #

def build_model(algo, vec_env, log_dir):
    algo = algo.lower()

    if algo == "dqn":
        model = DQN(
            "MlpPolicy",
            vec_env,
            learning_rate=0.0003,
            buffer_size=50_000,
            batch_size=64,
            exploration_fraction=0.2,
            exploration_final_eps=0.02,
            target_update_interval=2000, 
            gamma=0.99,
            optimize_memory_usage=False,
            tensorboard_log=log_dir,
            verbose=1,
        )

    elif algo == "ddqn":
        model = DDQN(
            "MlpPolicy",
            vec_env,
            learning_rate=0.0003,
            buffer_size=50_000,
            batch_size=64,
            exploration_fraction=0.2,
            exploration_final_eps=0.02,
            target_update_interval=2000,
            gamma=0.99,
            optimize_memory_usage=False,
            tensorboard_log=log_dir,
            verbose=1,
        )

    elif algo == "ppo":
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=0.0003,
            n_steps=2_048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log=log_dir,
            verbose=1,
        )

    else:
        raise ValueError(f"Unknown algorithm '{algo}'. Choose from: dqn, ddqn, ppo")

    return model


# ═══════════════════════════════════════════════════════════════════════════ #
#  DDA evaluation (frozen policy)                                            #
# ═══════════════════════════════════════════════════════════════════════════ #

def run_dda_only(model, env, dda, n_episodes=500, log_dir=None, base_seed=42, 
                 spike_episode=None):
    """
    Evaluate a frozen policy under live DDA control for n_episodes.
    """
    spike_applied = False
    pre_spike_peak = None
    recovery_threshold = None
    t_drop = None
    t_recovery = None
    adaptation_latency = None
    recovery_check_start = None

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
            
            # Spike injection
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
                      f"speed→{dda.max_speed} ***")
                print(f"*** Recovery check will start at episode {recovery_check_start} ***\n")

            # Recovery detection
            if (spike_applied and t_recovery is None and 
                recovery_check_start is not None and 
                episode_count >= recovery_check_start):
                if rolling_avg >= recovery_threshold:
                    t_recovery = episode_count
                    adaptation_latency = t_recovery - t_drop
                    print(f"*** RECOVERY at ep {t_recovery} | "
                          f"latency={adaptation_latency} eps ***")
                    
            if log_dir:
                writer.add_scalar("dda/current_speed", dda.current_speed, episode_count)
                writer.add_scalar("dda/episode_reward", ep_reward, episode_count)
                writer.add_scalar("dda/rolling_avg_reward", rolling_avg, episode_count)

            print(
                f"Episode {episode_count:4d} | reward: {ep_reward:6.0f} | "
                f"speed: {dda.current_speed:.2f} | rolling_avg: {rolling_avg:.0f}"
            )

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


# ═══════════════════════════════════════════════════════════════════════════ #
#  DDA callback                                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

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
                self.logger.record(
                    "dda/rolling_avg_reward",
                    np.mean(list(self.dda.reward_buffer))
                )
            self.logger.record("dda/current_speed", self.dda.current_speed)
            self.logger.record("dda/episode_reward", self.ep_reward)
            self.ep_reward = 0.0

        return True


# ═══════════════════════════════════════════════════════════════════════════ #
#  Main entry point                                                           #
# ═══════════════════════════════════════════════════════════════════════════ #

def main(algo, env_type, resume_from, delta, seed, timesteps, output_dir,
         max_speed=9.0, spike_episode=None, ablation=None):
    """
    Orchestrates training or DDA evaluation depending on --resume flag.
    """
    run_id = f"{env_type}_{algo}_delta{str(delta).replace('.','p')}_seed{seed}"
    if spike_episode is not None:
        run_id += "_spike"
    
    if ablation is not None:
        run_id += f"_abl_{ablation}"
    
    log_dir = os.path.abspath(os.path.join(output_dir, run_id))
    os.makedirs(log_dir, exist_ok=True)

    env = make_env(env_type=env_type, initial_speed=2.0)

    # ── Evaluation mode ────────────────────────────────────────────────── #
    if resume_from is not None:
        print(f"\n{'='*60}")
        print(f"EVALUATION MODE")
        print(f"  Algorithm   : {algo.upper()}")
        print(f"  Environment : {env_type}")
        print(f"  Model       : {resume_from}")
        print(f"  Delta       : {delta}  |  Seed : {seed}")
        print(f"  Max Speed   : {max_speed}")
        print(f"  Spike       : {spike_episode if spike_episode else 'Disabled'}")
        if ablation is not None:
            print(f"  🔬 Ablation  : {ablation}")
        print(f"{'='*60}\n")

        if algo.lower() == "ddqn":
            model = DDQN.load(resume_from)
        elif algo.lower() == "dqn":
            model = DQN.load(resume_from)
        elif algo.lower() == "ppo":
            model = PPO.load(resume_from)
        else:
            raise ValueError(f"Unknown algorithm: {algo}")

        # Flow zone is calibrated per environment.
        flow_low, flow_high = FLOW_ZONES[env_type]["default"]
        dda = DDAController(
                window_size=10, flow_low=flow_low, flow_high=flow_high,
                delta=delta, min_speed=2.0, max_speed=max_speed, initial_speed=2.0
            )

        # ── 🔬 APPLY ABLATION MODIFICATIONS ────────────────────────────────── #
        if ablation is not None:
            if ablation == "no_dda":
                # Complete disable: lock speed at max_speed (fixed challenge speed).
                # CRITICAL FIX: dda.update must still record episode_rewards and
                # rolling_avg_history (so FZC/mean reward can be computed
                # downstream) — it must only skip the SPEED ADJUSTMENT logic.
                # Previously this was a pure no-op, leaving episode_rewards
                # empty and causing np.mean() on an empty list -> nan.
                fixed_speed = max_speed
                env.unwrapped.set_difficulty(fixed_speed)
                dda.current_speed = fixed_speed

                def no_dda_update(reward):
                    dda.episode_rewards.append(reward)
                    dda.reward_buffer.append(reward)
                    if len(dda.reward_buffer) >= dda.window_size:
                        avg = float(np.mean(dda.reward_buffer))
                    else:
                        avg = float(reward)
                    dda.rolling_avg_history.append(avg)
                    dda.speed_history.append(dda.current_speed)
                    # current_speed intentionally NOT modified

                dda.update = no_dda_update

                def no_dda_set_difficulty(speed):
                    # Ignore all external speed-change requests
                    pass
                env.unwrapped.set_difficulty = no_dda_set_difficulty
                env.unwrapped.enemy_speed = fixed_speed
                print(f"  🔬 Ablation: No DDA (fixed speed at {fixed_speed})")

            elif ablation == "fixed_difficulty":
                # DDA Controller is "active" but speed is held constant at min_speed.
                # Same fix as above: must still record metrics, just never
                # change speed.
                env.unwrapped.set_difficulty(dda.min_speed)
                dda.current_speed = dda.min_speed

                def fixed_update(reward):
                    dda.episode_rewards.append(reward)
                    dda.reward_buffer.append(reward)
                    if len(dda.reward_buffer) >= dda.window_size:
                        avg = float(np.mean(dda.reward_buffer))
                    else:
                        avg = float(reward)
                    dda.rolling_avg_history.append(avg)
                    dda.speed_history.append(dda.current_speed)

                dda.update = fixed_update

                def fixed_set_difficulty(speed):
                    pass
                env.unwrapped.set_difficulty = fixed_set_difficulty
                print(f"  🔬 Ablation: Fixed Difficulty (speed held constant at {dda.min_speed})")

            elif ablation == "no_rolling_window":
                dda.window_size = 1
                original_update = dda.update
                def no_rolling_update(reward):
                    dda.reward_buffer.clear()
                    dda.reward_buffer.append(reward)
                    original_update(reward)
                dda.update = no_rolling_update
                print("  🔬 Ablation: No Rolling Window (window_size=1, immediate reward)")

            elif ablation == "no_smoothing":
                dda.window_size = 1
                original_update = dda.update
                def no_smoothing_update(reward):
                    dda.reward_buffer.clear()
                    dda.reward_buffer.append(reward)
                    original_update(reward)
                dda.update = no_smoothing_update
                print("  🔬 Ablation: No Reward Smoothing (raw rewards)")
            
            else:
                print(f"  ⚠️ Unknown ablation mode: {ablation}")
        # ────────────────────────────────────────────────────────────────────── #

        start_time = time.time()
        latency_result = run_dda_only(
            model, env, dda, n_episodes=500,
            log_dir=log_dir, base_seed=seed,
            spike_episode=spike_episode
        )
        wall_time = time.time() - start_time

        # Save arrays
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

    # ── Training mode ──────────────────────────────────────────────────── #
    print(f"\n{'='*60}")
    print(f"TRAINING MODE")
    print(f"  Algorithm   : {algo.upper()}")
    print(f"  Environment : {env_type}")
    print(f"  Timesteps   : {timesteps:,}")
    print(f"{'='*60}\n")

    vec_env = DummyVecEnv([lambda: make_env(env_type=env_type)])
    model = build_model(algo, vec_env, log_dir)

    eval_env = DummyVecEnv([lambda: make_env(env_type=env_type)])
    checkpoint_cb = CheckpointCallback(
        save_freq=100_000,
        save_path=log_dir,
        name_prefix="checkpoint"
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=50_000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1
    )

    start_time = time.time()
    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_cb, eval_cb],
        tb_log_name=run_id
    )
    wall_time = time.time() - start_time

    model_path = os.path.join(log_dir, "final_model")
    model.save(model_path)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE  |  {run_id}")
    print(f"  Wall-clock time : {wall_time:.1f}s  ({wall_time/60:.1f} min)")
    print(f"  Model saved to  : {model_path}.zip")
    print(f"  Best model at   : {os.path.join(log_dir, 'best_model.zip')}")
    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════════════ #
#  CLI                                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or evaluate DQN / DDQN / PPO on ChaseEnv or MultiChaseEnv"
    )
    parser.add_argument("--algo",      type=str,   default="ddqn",
                        choices=["dqn", "ddqn", "ppo"],
                        help="RL algorithm to use")
    parser.add_argument("--env",       type=str,   default="single",
                        choices=["single", "multi"],
                        help="Environment variant")
    parser.add_argument("--resume",    type=str,   default=None,
                        help="Path to a saved .zip model (triggers evaluation mode)")
    parser.add_argument("--delta",     type=float, default=0.05,
                        help="DDA step size (evaluation mode only)")
    parser.add_argument("--seed",      type=int,   default=42,
                        help="Base random seed (evaluation mode only)")
    parser.add_argument("--timesteps", type=int,   default=1_000_000,
                        help="Training timesteps (training mode only)")
    parser.add_argument("--output",    type=str,   default="./logs/experiments/",
                        help="Root directory for logs and saved models")
    parser.add_argument("--max_speed", type=float, default=9.0,
                        help="Maximum enemy speed for DDA Controller (evaluation mode only)")
    parser.add_argument("--spike",     type=int,   default=None,
                        help="Episode number to inject difficulty spike (e.g., 250). If omitted, no spike.")
    
    # ── 🔬 Ablation argument ────────────────────────────────────────────── #
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["no_dda", "fixed_difficulty", "no_rolling_window", "no_smoothing"],
                        help="Ablation mode (evaluation mode only): no_dda, fixed_difficulty, no_rolling_window, no_smoothing")
    # ──────────────────────────────────────────────────────────────────────── #

    args = parser.parse_args()

    main(
        algo=args.algo,
        env_type=args.env,
        resume_from=args.resume,
        delta=args.delta,
        seed=args.seed,
        timesteps=args.timesteps,
        output_dir=args.output,
        max_speed=args.max_speed,
        spike_episode=args.spike,
        ablation=args.ablation
    )