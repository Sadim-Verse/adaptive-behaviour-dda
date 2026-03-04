import pygame
from game_engine import GameEngine

 # Initialize Pygame for visualization
pygame.init()
screen = pygame.display.set_mode((800, 400))
clock = pygame.time.Clock()

game = GameEngine()
running = True

print("Press LEFT/RIGHT to move. ESC to quit")

while running:
    # Handle Human Input
    action = 0  # Default stay
    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT]:
        action = 1
    elif keys[pygame.K_RIGHT]:
        action = 2

    for event in pygame.event.get():
        if event.type == pygame.QUIT or keys[pygame.K_ESCAPE]:
            running = False
        # TEST DIFFICULTY: Press UP to increase difficulty
        if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
            game.set_difficulty(game.enemy_speed + 2)
            print(f"Difficulty increased to {game.enemy_speed}")
        # elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
        #     game.set_difficulty(game.enemy_speed - 2)

    # Update Game Physics
    state, reward, done = game.step(action)

    # Draw
    game.render(screen)

    # Handle Reset
    if done:
        print(f"Game Over! Final Score: {game.score}")
        game.reset()

    clock.tick(60)  # Lock to 60fps for human eyes

pygame.quit()