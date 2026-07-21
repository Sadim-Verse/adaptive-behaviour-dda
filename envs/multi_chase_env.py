import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class MultiChaseEnv(gym.Env):
    """
    Dynamic Multi-Enemy Chase Environment.

    Three homing enemies fall toward the player simultaneously.
    Each enemy has its own spawn position, speed multiplier, and
    horizontal homing strength, introducing genuine stochasticity
    into both spawn patterns and movement dynamics.

    Observation space (8D, normalised to [0, 1]):
        [0] player_x
        [1] enemy1_x
        [2] enemy1_y
        [3] enemy2_x
        [4] enemy2_y
        [5] enemy3_x
        [6] enemy3_y
        [7] min_enemy_y  (proximity of nearest enemy — a danger signal)

    Action space:
        Discrete(3): 0 = left, 1 = right, 2 = stay

    Difficulty variable: base_speed — adjusted by DDA Controller via set_difficulty().
    Each enemy's actual speed = base_speed * individual speed multiplier (0.8–1.2),
    so enemies are not perfectly synchronised, adding realistic stochasticity.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    N_ENEMIES = 3

    def __init__(self, render_mode=None, enemy_speed=2.0, homing_strength=3.0):
        super().__init__()

        self.width = 400
        self.height = 400
        self.player_size = 15
        self.enemy_size = 20
        self.max_steps = 2000

        # Base difficulty — adjusted by DDA Controller
        self.enemy_speed = enemy_speed
        self.homing_strength = homing_strength

        # Per-enemy speed multipliers — randomised on reset for stochasticity
        self._speed_multipliers = np.ones(self.N_ENEMIES)

        # Action space: left / right / stay
        self.action_space = spaces.Discrete(3)

        # 8D normalised observation
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )

        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.current_step = 0

        # Enemy state arrays
        self.enemy_x = np.zeros(self.N_ENEMIES)
        self.enemy_y = np.zeros(self.N_ENEMIES)

    # ------------------------------------------------------------------ #
    #  Core Gymnasium API                                                  #
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Player starts at centre bottom
        self.player_x = float(self.width // 2)

        # Enemies spawn at staggered vertical positions so they do not all
        # arrive simultaneously — this creates a harder, more varied challenge
        for i in range(self.N_ENEMIES):
            self.enemy_x[i] = float(
                self.np_random.integers(self.enemy_size, self.width - self.enemy_size)
            )
            # Stagger starting y: enemy 0 at top, enemy 1 and 2 further back
            self.enemy_y[i] = float(-i * (self.height // self.N_ENEMIES))

        # Randomise per-enemy speed multipliers (±20% of base speed)
        self._speed_multipliers = self.np_random.uniform(0.8, 1.2, size=self.N_ENEMIES)

        self.current_step = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        # --- Player movement ---
        move = 8
        if action == 0:
            self.player_x = max(self.player_size, self.player_x - move)
        elif action == 1:
            self.player_x = min(self.width - self.player_size, self.player_x + move)
        # action 2: stay

        # --- Enemy movement ---
        collision = False
        for i in range(self.N_ENEMIES):
            # Homing: shift x toward player
            dx = self.player_x - self.enemy_x[i]
            shift = np.clip(dx, -self.homing_strength, self.homing_strength)
            self.enemy_x[i] += shift
            self.enemy_x[i] = np.clip(
                self.enemy_x[i], self.enemy_size, self.width - self.enemy_size
            )

            # Fall at base speed × individual multiplier
            self.enemy_y[i] += self.enemy_speed * self._speed_multipliers[i]

            # Collision check
            if self._check_collision_single(i):
                collision = True

            # Respawn this enemy when it exits the bottom
            if self.enemy_y[i] > self.height:
                self.enemy_y[i] = float(
                    self.np_random.integers(-self.height // 2, -self.enemy_size)
                )
                self.enemy_x[i] = float(
                    self.np_random.integers(self.enemy_size, self.width - self.enemy_size)
                )
                # Randomise this enemy's multiplier on respawn
                self._speed_multipliers[i] = self.np_random.uniform(0.8, 1.2)

        if collision:
            reward = -100.0
            terminated = True
        else:
            terminated = False
            # Base survival reward
            reward = 1.0

            # Distance bonus — reward agent for staying away from nearest active enemy
            distances = []
            for i in range(self.N_ENEMIES):
                if self.enemy_y[i] > 0:
                    dx = abs(self.player_x - self.enemy_x[i])
                    dy = abs((self.height - self.player_size - 5) - self.enemy_y[i])
                    distances.append(float(np.sqrt(dx**2 + dy**2)))

            if distances:
                min_dist = min(distances)
                # Normalised bonus: max ~0.5 when far, ~0 when close
                # Scaled small enough not to dominate survival reward
                distance_bonus = (min_dist / (self.width * 1.414)) * 0.5
                reward += distance_bonus

        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ------------------------------------------------------------------ #
    #  Observation / Info                                                  #
    # ------------------------------------------------------------------ #

    def _get_obs(self):
        # Closest enemy y (normalised) as danger proximity signal
        visible_y = self.enemy_y[self.enemy_y > 0]
        min_y = float(np.min(visible_y)) if len(visible_y) > 0 else 0.0

        obs = np.array([
            self.player_x / self.width,          # [0] player x
            self.enemy_x[0] / self.width,        # [1] enemy 1 x
            max(0.0, self.enemy_y[0]) / self.height,  # [2] enemy 1 y
            self.enemy_x[1] / self.width,        # [3] enemy 2 x
            max(0.0, self.enemy_y[1]) / self.height,  # [4] enemy 2 y
            self.enemy_x[2] / self.width,        # [5] enemy 3 x
            max(0.0, self.enemy_y[2]) / self.height,  # [6] enemy 3 y
            min_y / self.height,                 # [7] nearest enemy proximity
        ], dtype=np.float32)
        return obs

    def _get_info(self):
        return {"enemy_speed": self.enemy_speed}

    # ------------------------------------------------------------------ #
    #  Collision                                                           #
    # ------------------------------------------------------------------ #

    def _check_collision_single(self, i):
        player_rect = pygame.Rect(
            self.player_x - self.player_size // 2,
            self.height - self.player_size - 5,
            self.player_size, self.player_size
        )
        enemy_rect = pygame.Rect(
            self.enemy_x[i] - self.enemy_size // 2,
            self.enemy_y[i] - self.enemy_size // 2,
            self.enemy_size, self.enemy_size
        )
        return player_rect.colliderect(enemy_rect)

    # ------------------------------------------------------------------ #
    #  DDA Interface                                                       #
    # ------------------------------------------------------------------ #

    def set_difficulty(self, new_speed):
        """Called by the DDA Controller to adjust base enemy fall speed."""
        self.enemy_speed = float(np.clip(new_speed, 0.3, 9.0))

    # ------------------------------------------------------------------ #
    #  Rendering                                                           #
    # ------------------------------------------------------------------ #

    def render(self):
        if self.render_mode != "human":
            return
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

        self.window.fill((20, 20, 20))

        # Player — green
        pygame.draw.rect(
            self.window, (0, 255, 0),
            (int(self.player_x - self.player_size // 2),
             self.height - self.player_size - 5,
             self.player_size, self.player_size)
        )

        # Enemies — shades of red/orange to distinguish them
        colours = [(255, 60, 60), (255, 150, 0), (255, 80, 200)]
        for i in range(self.N_ENEMIES):
            if self.enemy_y[i] > 0:
                pygame.draw.rect(
                    self.window, colours[i],
                    (int(self.enemy_x[i] - self.enemy_size // 2),
                     int(self.enemy_y[i] - self.enemy_size // 2),
                     self.enemy_size, self.enemy_size)
                )

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
