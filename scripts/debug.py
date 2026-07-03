import numpy as np
from stable_baselines3 import DQN
from envs.chase_env import ChaseEnv
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor

# Load the perfect model
model = DQN.load("logs/chase_dda/perfect_model_backup.zip")

# Build the environment EXACTLY as training does
def make_env(initial_speed=2.0):
    env = ChaseEnv(enemy_speed=initial_speed, homing_strength=3.0)
    env = TimeLimit(env, max_episode_steps=2000)
    env = Monitor(env, info_keywords=('enemy_speed',))
    return env

env = make_env(initial_speed=2.0)

# Run one deterministic episode
obs, _ = env.reset()
total_reward = 0
for step in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print(f"Result: survived {step+1} steps, total reward {total_reward}")