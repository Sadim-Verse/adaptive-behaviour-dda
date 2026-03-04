import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from my_game_logic import GameState     # pure python game logic

class AdaptiveGameEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(AdaptiveGameEnv, self).__init__()

        # Define Action Space (Left, Right, Stay)
        self.action_space = spaces.Discrete(3)

        # Define Observation Space
        # (e.g., Player X, Enemy X, Enemy Y, Enemy Speed)
        # We normalize values between 0 and 1 for better AI performance
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

        self.game_state = GameState()
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        observation = self.game_state.reset_game()
        return observation, {}
    
    def step(self, action):
        # Apply action to the game engine
        self.game_state.update(action)

        # Get the new state
        observation = self.game_state.get_state()

        # Calculate Reward (important part for convergence)
        # Reward survival, penalizes death heavily
        if self.game_state.is_alive:
            reward = 1.0
        else:
            reward = -100.0

        terminated = not self.game_state.is_alive
        truncated = False   # used if time runs out
        
        return observation, reward, terminated, truncated, {}
    
    def set_difficulty(self, level):
        """The crucial function"""
        self.game_state.change_speed(level)