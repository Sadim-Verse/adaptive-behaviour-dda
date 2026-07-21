import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class ChaseEnv(gym.Env):
    """
    A single homing enemy that falls toward the player.
    The player must dodge left or right to avoid collision.
    Standing still is fatal because the enemy homes in on the player's x-position.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, enemy_speed=2.0, homing_strength=3.0):
        super().__init__()
        self.width = 400
        self.height = 400
        self.player_size = 15
        self.enemy_size = 20

        # Enemy dynamics: vertical fall speed, horizontal homing strength
        self.enemy_speed = enemy_speed
        self.homing_strength = homing_strength
        self.max_steps = 2000          # episode terminates after this many steps

        # 3 discrete actions: 0 = left, 1 = right, 2 = stay
        self.action_space = spaces.Discrete(3)

        # Observation: normalised player x, enemy x, enemy y
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.current_step = 0

    def reset(self, seed=None, options=None):
        """Start a new episode. Place player at centre, enemy at random x at top."""
        super().reset(seed=seed)
        self.player_x = self.width // 2
        self.enemy_x = self.np_random.integers(self.enemy_size, self.width - self.enemy_size)
        self.enemy_y = 0
        self.current_step = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        """Execute one timestep: move player, move enemy, check collision, compute reward."""
        move = 8
        if action == 0:               # left
            self.player_x = max(self.player_size, self.player_x - move)
        elif action == 1:             # right
            self.player_x = min(self.width - self.player_size, self.player_x + move)
        # action 2: stay – no horizontal movement

        # Homing enemy: adjust x towards player, then fall
        dx = self.player_x - self.enemy_x
        shift = np.clip(dx, -self.homing_strength, self.homing_strength)
        self.enemy_x += shift
        self.enemy_x = np.clip(self.enemy_x, self.enemy_size, self.width - self.enemy_size)
        self.enemy_y += self.enemy_speed

        collision = self._check_collision()
        reward = 1.0                   # +1 for every frame survived
        terminated = False
        if collision:
            reward = -100.0            # heavy penalty for dying
            terminated = True

        # Respawn enemy when it goes off the bottom
        if self.enemy_y > self.height:
            self.enemy_y = 0
            self.enemy_x = self.np_random.integers(self.enemy_size, self.width - self.enemy_size)

        self.current_step += 1
        truncated = (self.current_step >= self.max_steps)

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self):
        """Return the normalised observation vector."""
        return np.array([self.player_x / self.width,
                         self.enemy_x / self.width,
                         self.enemy_y / self.height], dtype=np.float32)

    def _get_info(self):
        """Additional info, used by the Monitor wrapper for logging."""
        return {"enemy_speed": self.enemy_speed}

    def _check_collision(self):
        """Return True if the player rectangle overlaps the enemy rectangle."""
        player_rect = pygame.Rect(self.player_x - self.player_size//2,
                                 self.height - self.player_size - 5,
                                 self.player_size, self.player_size)
        enemy_rect = pygame.Rect(self.enemy_x - self.enemy_size//2,
                                self.enemy_y - self.enemy_size//2,
                                self.enemy_size, self.enemy_size)
        return player_rect.colliderect(enemy_rect)

    def set_difficulty(self, new_speed):
        """Called by the DDA controller to adjust the enemy fall speed."""
        self.enemy_speed = np.clip(new_speed, 0.3, 9.0)

    def render(self):
        """Draw the game frame using Pygame (only in 'human' mode)."""
        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                self.window = pygame.display.set_mode((self.width, self.height))
                self.clock = pygame.time.Clock()
            self.window.fill((20, 20, 20))
            # Player as green rectangle
            pygame.draw.rect(self.window, (0, 255, 0),
                            (self.player_x - self.player_size//2,
                             self.height - self.player_size - 5,
                             self.player_size, self.player_size))
            # Enemy as red rectangle
            pygame.draw.rect(self.window, (255, 0, 0),
                            (self.enemy_x - self.enemy_size//2,
                             self.enemy_y - self.enemy_size//2,
                             self.enemy_size, self.enemy_size))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        """Clean up Pygame resources."""
        if self.window is not None:
            pygame.quit()
            self.window = None