import pygame
import random
import numpy as np

# Constants
WIDTH = 800
HEIGHT = 400
PLAYER_SIZE = 40
ENEMY_SIZE = 30
PLAYER_SPEED = 10   # How fast the agent moves

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

class GameEngine:
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT

        # Difficulty parameters
        self.enemy_speed = 2    # Initial speed (level 1)
        self.spawn_rate = 20    # How often enemies spawn (frames)

        # Game State
        self.player_x = self.width // 2
        self.player_y = self.height - PLAYER_SIZE - 10
        self.enemies = []   # List of [x, y] coordinates
        self.score = 0
        self.game_over = False
        self.clock_tick = 0 # To track spawning

    def reset(self):
        # Resets the game to the starting state.
        self.player_x = self.width // 2
        self.enemies = []
        self.score = 0
        self.game_over = False
        self.clock_tick = 0
        
        # Return the initial state
        return self.get_state()
    
    def set_difficulty(self, speed_level):
        # Core Function. Adjust physics parameters dynamically.
        self.enemy_speed = speed_level
        # Make enemies spawn faster as speed increases
        self.spawn_rate = max(5, 30 - int(speed_level))

    def step(self, action):
        # Advances the game logic by one frame. action: 0 = Stay, 1 = Left, 2 = Right
        if self.game_over:
            return self.get_state(), 0, True # State, Reward, Done

        # Move Player
        if action == 1: # Left
            self.player_x -= PLAYER_SPEED
        elif action == 2: # Right
            self.player_x += PLAYER_SPEED
            
        # Clamp player to screen
        self.player_x = max(0, min(self.player_x, self.width - PLAYER_SIZE))

        # Spawn Enemies
        self.clock_tick += 1
        if self.clock_tick >= self.spawn_rate:
            # 70% chance to target player, 30% random
            if random.random() < 0.5:
                # Spawn roughly above the player
                offset = random.randint(-20, 20)    # small random offset
                enemy_x = self.player_x + offset
            else:
                # Random spawn to keep it unpredictable
                enemy_x = random.randint(0, self.width - ENEMY_SIZE)

            # Ensure Enemy stays on screen
            enemy_x = max(0, min(enemy_x,self.width - ENEMY_SIZE))

            self.enemies.append([enemy_x, 0])
            self.clock_tick = 0

        # Move Enemies & Check Collisions
        reward = 1 # Survival reward
        
        # We iterate backwards to safely remove items from the list
        for enemy in self.enemies[:]:
            enemy[1] += self.enemy_speed
            
            # Check Collision (Simple AABB - Axis Aligned Bounding Box)
            if (enemy[1] + ENEMY_SIZE >= self.player_y and 
                enemy[1] <= self.player_y + PLAYER_SIZE and
                enemy[0] + ENEMY_SIZE >= self.player_x and 
                enemy[0] <= self.player_x + PLAYER_SIZE):
                
                self.game_over = True
                reward = -100 # Death penalty
                break

            # Remove enemies that hit the floor
            if enemy[1] > self.height:
                self.enemies.remove(enemy)
                self.score += 1 # Bonus for dodging successfully

        return self.get_state(), reward, self.game_over
    
    def get_state(self):
        """
        Returns the raw numbers the AI needs to 'see'.
        We return: [Player X, Closest Enemy X, Closest Enemy Y]
        We normalize these numbers (0.0 to 1.0) because Neural Networks love small numbers.
        """
        # Find closest enemy (if any)
        closest_enemy_x = 0
        closest_enemy_y = 0
        
        # Focus on enemies closest to the bottom
        if len(self.enemies) > 0:
            # Sort by Y descending (highest Y = closest to bottom/player)
            most_dangerous_enemy = max(self.enemies, key=lambda e: e[1])
            closest_enemy_x = most_dangerous_enemy[0] / self.width
            closest_enemy_y = most_dangerous_enemy[1] / self.height
        
        norm_player_x = self.player_x / self.width
        
        return np.array([norm_player_x, closest_enemy_x, closest_enemy_y], dtype=np.float32)

    def render(self, screen):
        """
        Draws the state to the Pygame window.
        Separated from step() so we can train without rendering.
        """
        screen.fill(WHITE)
        
        # Draw Player
        pygame.draw.rect(screen, BLUE, (self.player_x, self.player_y, PLAYER_SIZE, PLAYER_SIZE))
        
        # Draw Enemies
        for enemy in self.enemies:
            pygame.draw.rect(screen, RED, (enemy[0], enemy[1], ENEMY_SIZE, ENEMY_SIZE))
            
        # Draw Score
        font = pygame.font.SysFont(None, 36)
        text = font.render(f"Score: {self.score}", True, (0, 0, 0))
        screen.blit(text, (10, 10))
        
        pygame.display.flip()
