import os
import argparse
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import TimeLimit
from envs.chase_env import ChaseEnv
from training.dda_controller import DDAController

def make_env(initial_speed=2.0):
    """
    Create and wrap the Chase environment.
    - TimeLimit caps episodes at 2000 steps.
    - Monitor logs episode statistics (reward, length) and extra info (enemy_speed).
    """
    env = ChaseEnv(enemy_speed=initial_speed, homing_strength=3.0)
    env = TimeLimit(env, max_episode_steps=2000)
    env = Monitor(env, info_keywords=('enemy_speed',))
    return env

def run_dda_only(model, env, dda, n_episodes=500, log_dir=None, base_seed=42):
    """
    Run the frozen model with the DDA controller, without any training.
    Configured to handle variable base seeds for statistical validation.
    """
    if log_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=log_dir)

    # Use the passed-in base seed
    obs, _ = env.reset(seed=base_seed)
    ep_reward = 0
    episode_count = 0

    while episode_count < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_reward += reward

        if terminated or truncated:
            dda.update(ep_reward)
            env.unwrapped.set_difficulty(dda.current_speed)
            episode_count += 1

            if log_dir:
                writer.add_scalar("dda/current_speed", dda.current_speed, episode_count)
                writer.add_scalar("dda/episode_reward", ep_reward, episode_count)
                if len(dda.reward_buffer) > 0:
                    writer.add_scalar("dda/rolling_avg_reward",
                                      np.mean(list(dda.reward_buffer)), episode_count)

            if len(dda.reward_buffer) > 0:
                print(f"Episode {episode_count:4d} | reward: {ep_reward:4.0f} | "
                      f"speed: {dda.current_speed:.2f} | "
                      f"rolling_avg: {np.mean(list(dda.reward_buffer)):.0f}")
            else:
                print(f"Episode {episode_count:4d} | reward: {ep_reward:4.0f} | "
                      f"speed: {dda.current_speed:.2f}")

            # Keep shifting environmental seed predictably based on the base_seed
            obs, _ = env.reset(seed=base_seed + episode_count)
            ep_reward = 0

    if log_dir:
        writer.close()

def train_chase_dda(timesteps=1_000_000, log_dir="./logs/chase_dda/", resume_from=None, delta=0.05, seed=42):
    """
    Main entry point parameterized for dynamic delta and seed distribution.
    """
    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    env = make_env(initial_speed=2.0)

    if resume_from is not None:
        print(f"Loading perfect model from {resume_from} | Delta: {delta} | Seed: {seed}")
        model = DQN.load(resume_from)

        # Parameterized delta
        dda = DDAController(window_size=10, flow_low=600.0, flow_high=1200.0,
                            delta=delta, min_speed=2, max_speed=6.0, initial_speed=2.0)
        
        print("Starting DDA-only run (no training)...")
        run_dda_only(model, env, dda, n_episodes=500, log_dir=log_dir, base_seed=seed)

        # Dynamic filenames to stop overwrites
        np.save(os.path.join(log_dir, f"rewards_delta_{delta}_seed_{seed}.npy"), dda.episode_rewards)
        np.save(os.path.join(log_dir, f"speeds_delta_{delta}_seed_{seed}.npy"), dda.speed_history)
        return model, dda

    else:
        # Code stays identical for clean continuous fallback training paths
        vec_env = DummyVecEnv([lambda: env])
        model = DQN("MlpPolicy", vec_env,
                    learning_rate=0.0003, buffer_size=50000, batch_size=64,
                    exploration_fraction=0.2, exploration_final_eps=0.02,
                    target_update_interval=2000, gamma=0.99,
                    tensorboard_log=log_dir, verbose=1)

        dda = DDAController(window_size=10, flow_low=600.0, flow_high=1200.0,
                            delta=delta, min_speed=2, max_speed=6.0, initial_speed=2.0)

        class DDACallback(BaseCallback):
            def __init__(self, dda, env, verbose=0):
                super().__init__(verbose)
                self.dda = dda
                self.env = env
                self.ep_reward = 0

            def _on_step(self):
                self.ep_reward += self.locals['rewards'][0]
                if self.locals['dones'][0]:
                    self.dda.update(self.ep_reward)
                    unwrapped = self.env.unwrapped
                    if hasattr(unwrapped, 'set_difficulty'):
                        unwrapped.set_difficulty(self.dda.current_speed)
                    if len(self.dda.reward_buffer) > 0:
                        self.logger.record('dda/rolling_avg_reward', np.mean(list(self.dda.reward_buffer)))
                    self.logger.record('dda/current_speed', self.dda.current_speed)
                    self.logger.record('dda/episode_reward', self.ep_reward)
                    self.ep_reward = 0
                return True

        callback = DDACallback(dda, env)
        model.learn(total_timesteps=timesteps, callback=callback)
        model.save(os.path.join(log_dir, "dqn_chase_dda"))
        return model, dda

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to .zip model to run with DDA")
    parser.add_argument("--delta", type=float, default=0.05, help="Delta value for DDA adjustment")
    parser.add_argument("--seed", type=int, default=42, help="Stochastic base evaluation seed")
    args = parser.parse_args()
    
    # Save results directly inside an target directory
    output_directory = "./logs/multi_seed_eval/"
    
    model, dda = train_chase_dda(
        resume_from=args.resume, 
        delta=args.delta, 
        seed=args.seed, 
        log_dir=output_directory
    )
    print(f"Completed: Delta {args.delta} | Seed {args.seed} | Flow Zone Consistency: {dda.get_flow_zone_consistency():.2%}")