"""
Run this locally (not in a headless environment) to visually verify:
  - hazards actually home toward the player's x-position as they fall
  - hazards respawn at the top after passing the bottom row
  - the player's left/right/stay movement feels right
  - collision correctly ends the episode

Controls:
  A / Left arrow  -> move left
  D / Right arrow -> move right
  (no key held)   -> stay
  Q               -> quit

Usage:
  pip install minigrid pygame-ce --break-system-packages   # if not already installed
  python verify_hazard_behavior.py --difficulty 0.5
"""

import argparse
import sys
import pygame

from envs.obstacle_minigrid_env import ObstacleMiniGridEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", type=float, default=0.5,
                         help="Difficulty level in [0, 1] to test (default 0.5)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fps", type=int, default=10,
                         help="Lower = easier to watch hazard movement step by step")
    args = parser.parse_args()

    env = ObstacleMiniGridEnv(difficulty=args.difficulty, max_steps=2000, render_mode="human")
    obs, info = env.reset(seed=args.seed)
    print(f"Episode start. difficulty={info['difficulty']}, "
          f"num_hazards={info['num_hazards']}, vertical_speed={info['vertical_speed']}, "
          f"horizontal_speed={info['horizontal_speed']}, grid_size={info['grid_size']}")

    clock = pygame.time.Clock()
    total_reward = 0.0
    step = 0
    running = True

    while running:
        action = 2  # default: stay
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            running = False
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
            action = 0
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            action = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        env.render()
        clock.tick(args.fps)

        if terminated:
            print(f"COLLISION at step {step}. total_reward={total_reward:.1f}")
            print("Resetting for a new episode in 1s...")
            pygame.time.wait(1000)
            obs, info = env.reset()
            total_reward = 0.0
            step = 0
        elif truncated:
            print(f"Episode truncated (max_steps) at step {step}. total_reward={total_reward:.1f}")
            running = False

    env.close()
    sys.exit(0)


if __name__ == "__main__":
    main()