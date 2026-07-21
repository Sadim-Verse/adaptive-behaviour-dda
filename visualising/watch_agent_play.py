"""
Watch a trained model play ObstacleMiniGridEnv, live, in a rendered window.

Two modes:
  1. Fixed difficulty (default) -- watch raw policy behavior at one
     difficulty level, no DDA involved. Useful for sanity-checking what
     the agent actually learned to do.
  2. --dda -- watch the SAME frozen model under live DDA Controller
     adjustment, same as what run_dda_only() does numerically, but
     rendered so you can see the difficulty change happening in real time.

Usage:
  python watch_agent_play.py --algo ppo --model logs/procedural_env/obstacle_minigrid_ppo_delta0p05_seed42/best_model.zip --difficulty 0.3

  python watch_agent_play.py --algo ppo --model .../best_model.zip --dda --delta 0.05 --max_speed 1.0
"""

import argparse
import time
import numpy as np
import pygame

from stable_baselines3 import DQN, PPO
from training.ddqn import DDQN
from envs.obstacle_minigrid_env import ObstacleMiniGridEnv


def load_model(algo, path):
    algo = algo.lower()
    if algo == "dqn":
        return DQN.load(path)
    elif algo == "ddqn":
        return DDQN.load(path)
    elif algo == "ppo":
        return PPO.load(path)
    raise ValueError(f"Unknown algorithm '{algo}'. Choose from: dqn, ddqn, ppo")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True, choices=["dqn", "ddqn", "ppo"])
    parser.add_argument("--model", type=str, required=True,
                        help="Path to a saved .zip model (best_model.zip or final_model.zip)")
    parser.add_argument("--difficulty", type=float, default=0.3,
                        help="Fixed difficulty level in [0,1] (ignored if --dda is set)")
    parser.add_argument("--dda", action="store_true",
                        help="Enable live DDA Controller instead of a fixed difficulty")
    parser.add_argument("--delta", type=float, default=0.05, help="DDA step size (--dda mode only)")
    parser.add_argument("--max_speed", type=float, default=1.0, help="DDA max difficulty (--dda mode only)")
    parser.add_argument("--min_speed", type=float, default=0.0, help="DDA min difficulty (--dda mode only)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to watch before exiting")
    args = parser.parse_args()

    model = load_model(args.algo, args.model)
    print(f"Loaded {args.algo.upper()} model from {args.model}")

    initial_difficulty = args.min_speed if args.dda else args.difficulty
    env = ObstacleMiniGridEnv(difficulty=initial_difficulty, max_steps=2000, render_mode="human")

    dda = None
    if args.dda:
        from training.dda_controller import DDAController
        # NOTE: Flow Zone bounds for obstacle_minigrid are still a
        # placeholder (see FLOW_ZONES in train_multi_env.py) -- the
        # controller will run and visibly move the difficulty around, but
        # its target zone is not yet calibrated for this environment, so
        # don't read anything quantitative into WHERE it settles yet, only
        # whether the adjustment mechanism itself looks visually sane.
        dda = DDAController(window_size=10, flow_low=None, flow_high=None,
                            delta=args.delta, min_speed=args.min_speed,
                            max_speed=args.max_speed, initial_speed=initial_difficulty)
        print("DDA mode enabled -- Flow Zone bounds are still a placeholder for this "
              "environment, so treat this as a mechanism check, not a calibrated result.")

    clock = pygame.time.Clock()
    obs, info = env.reset(seed=args.seed)
    ep_reward = 0.0
    ep_count = 0
    step = 0
    running = True

    print(f"difficulty={info['difficulty']:.2f} num_hazards={info['num_hazards']} "
          f"v_speed={info['vertical_speed']:.2f} h_speed={info['horizontal_speed']:.2f}")

    while running and ep_count < args.episodes:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        ep_reward += reward
        step += 1
        env.render()
        clock.tick(args.fps)

        if terminated or truncated:
            ep_count += 1
            outcome = "DIED" if terminated else "SURVIVED (max_steps)"
            print(f"Episode {ep_count}/{args.episodes} | {outcome} | steps={step} | reward={ep_reward:.1f}", end="")

            if dda is not None:
                dda.update(ep_reward)
                env.set_difficulty(dda.current_speed)
                print(f" | new difficulty={dda.current_speed:.2f}")
            else:
                print()

            time.sleep(0.6)
            obs, info = env.reset(seed=None if args.seed is None else args.seed + ep_count)
            ep_reward = 0.0
            step = 0

    env.close()


if __name__ == "__main__":
    main()