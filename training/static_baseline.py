# scripts/static_baseline.py
"""
Run the trained DDQN agent at a fixed speed without DDA.
Prints the mean reward over a number of episodes.
Use this to compare against the DDA system's performance.
"""
from stable_baselines3 import DQN
from envs.chase_env import ChaseEnv

MODEL_PATH = "logs/chase_dqn/perfect_model_backup.zip"   # or perfect_model_backup.zip
SPEED = 4.15                                  # change to test different speeds
N_EPISODES = 500

model = DQN.load(MODEL_PATH)
env = ChaseEnv(render_mode=None, enemy_speed=SPEED, homing_strength=3.0)

total_reward = 0.0
for ep in range(N_EPISODES):
    obs, _ = env.reset(seed=42 + ep)
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

mean_reward = total_reward / N_EPISODES
print(f"Static baseline mean reward at speed {SPEED}: {mean_reward:.1f}")