import gymnasium as gym
from stable_baselines3 import DQN
from gym_wrapper import AdaptiveGameEnv
import pygame

# Load the Environment (With render_mode="human" to see it)
env = AdaptiveGameEnv(render_mode="human")

# Load the Trained Brain
model_path = r"C:\Users\HP\Projects\Adaptive_Agent\models\DQN\adaptive_agent_final.zip"
model = DQN.load(model_path, env=env)

print("Watching the Agent play...")

# Play Loop
obs, _ = env.reset()
for i in range(100000):
    # Ask the AI what to do
    action, _states = model.predict(obs, deterministic=True)

    # Do the action
    obs, reward, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()

env.close()