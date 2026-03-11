import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from game_engine import GameEngine

class AdaptiveGameEnv(gym.Env):
    """
    Custom Environment that follows gymnasium interface.
    This acts as the bridge between your GameEngine and the AI Agent.
    """
    metadata = {'render_modes': ['human', 'console']}

    def __init__(self, render_mode=None):
        super(AdaptiveGameEnv, self).__init__()

        # Initialize Game Engine
        self.game = GameEngine()

        # 2. Define Action Space: 3 discrete actions
        # 0 = Stay, 1 = Left, 2 = Right
        self.action_space = spaces.Discrete(3)

        # 3. Define Observation Space
        # We expect 3 numbers: [Player X, Enemy X, Enemy Y]
        # All normalized between 0 and 1
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(7,), 
            dtype=np.float32
        )

        # 4. Setup Rendering (Optional)
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.game.width, self.game.height))
            self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state and returns the initial observation.
        """
        # Seed for reproducibility
        super().reset(seed=seed)

        # Reset the game engine
        observation = self.game.reset()

        return observation, {}
    
    def step(self, action):
        """ Execute one time step within the environment. """
        # Advance the game logic
        observation, reward, done = self.game.step(action)

        # Handle Termination
        terminated = done # The agent died
        truncated = False # time limit

        # Render if needed
        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, {}
    
    def render(self):
        """
        Draws the game state to the screen.
        """
        if self.render_mode == "human":
            self.game.render(self.screen)
            # Cap the frame rate only if a human is watching
            # so we don't see a blur of 10,000 fps
            self.clock.tick(60)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            
    # --- The Thesis Function ---
    def set_difficulty(self, speed_level):
        """
        Allows the training script to dynamically adjust difficulty.
        """
        print(f"[Wrapper] Changing difficulty to speed: {speed_level}")
        self.game.set_difficulty(speed_level)
