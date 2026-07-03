# scripts/visualize_agent.py
"""
Interactive Agent Visualization Script.
Renders trained models (DQN/DDQN/PPO) inside 'single' or 'multi' environment profiles.
Supports both Fixed-Difficulty Baseline monitoring and Live DDA Control tracking.
"""
import os
import sys
import argparse
import pygame
import numpy as np
from stable_baselines3 import DQN, PPO

# Reuse your standardized environment factory and controller architectures
from training.train_multi_env import make_env
from training.dda_controller import DDAController

def run_visualization(algo, env_type, model_path, use_dda, fixed_speed, delta):
    print("=" * 70)
    print(f"🎬 INITIALIZING AGENT VISUALIZATION SCREEN")
    print(f"  Algorithm    : {algo.upper()}")
    print(f"  Environment  : {env_type.upper()}")
    print(f"  Model Source : {model_path}")
    print(f"  Control Mode : {'LIVE DDA ADAPTATION' if use_dda else 'STATIC BASELINE'}")
    if use_dda:
        print(f"  DDA Delta    : {delta}")
    else:
        print(f"  Fixed Speed  : {fixed_speed}")
    print("=" * 70)

    # 1. Verification of Model File Existences
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Target model file not found at: {model_path}")

    # 2. Dynamic Policy Class Instantiation
    if algo.lower() in ("dqn", "ddqn"):
        model = DQN.load(model_path)
    elif algo.lower() == "ppo":
        model = PPO.load(model_path)
    else:
        raise ValueError(f"Unsupported algorithmic configuration: {algo}")

    # 3. Environment Instantiation with Human Render Engine Enabled
    initial_speed = 2.0 if use_dda else fixed_speed
    env = make_env(env_type=env_type, initial_speed=initial_speed, render_mode="human")

    # 4. Conditional DDA Controller Instantiation
    dda = None
    if use_dda:
        # Match your exact thesis evaluation flow boundaries
        flow_low, flow_high = (80.0, 200.0) if env_type == "multi" else (600.0, 1200.0)
        dda = DDAController(
            window_size=10, 
            flow_low=flow_low, 
            flow_high=flow_high,
            delta=delta, 
            min_speed=2.0, 
            max_speed=6.0, 
            initial_speed=2.0
        )

    # Initialize Pygame window handles cleanly
    pygame.init()
    
    obs, _ = env.reset(seed=42)
    ep_reward = 0.0
    episode_count = 0
    running = True

    try:
        while running:
            # Prevent Pygame window from reporting "Not Responding" to the OS
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
            
            if not running:
                break

            # Execute deterministic inference with no exploration noise
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            
            # Explicit call to render step onto the viewport
            env.render()

            # Handle Episode Boundaries
            if terminated or truncated:
                episode_count += 1
                current_speed = env.unwrapped.enemy_speed if env_type == "single" else env.unwrapped.enemy_speed
                
                print(f"➔ Episode {episode_count:2d} Ended | Accumulated Reward: {ep_reward:6.1f} | Active Speed: {current_speed:.2f}")
                
                if use_dda and dda is not None:
                    dda.update(ep_reward)
                    env.unwrapped.set_difficulty(dda.current_speed)
                    rolling_avg = np.mean(list(dda.reward_buffer)) if len(dda.reward_buffer) > 0 else ep_reward
                    print(f"   [DDA ACTIVE] Updated Difficulty Speed Target ➔ {dda.current_speed:.2f} | 10-Ep Rolling Avg: {rolling_avg:.1f}")
                
                # Reset environment with changing structural variations via shifted seed keys
                obs, _ = env.reset(seed=42 + episode_count)
                ep_reward = 0.0

    except KeyboardInterrupt:
        print("\n⚠️ Visualization pipeline interrupted manually via command terminal.")
    
    finally:
        print("Closing visualization windows...")
        env.close()
        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch trained models navigate environments live.")
    parser.add_argument("--algo", type=str, required=True, choices=["dqn", "ddqn", "ppo"])
    parser.add_argument("--env", type=str, required=True, choices=["single", "multi"])
    parser.add_argument("--model", type=str, required=True, help="Path to best_model.zip or final_model.zip")
    parser.add_argument("--dda", action="store_true", help="Activate live DDA tracking adjustments")
    parser.add_argument("--speed", type=float, default=2.0, help="Fixed speed velocity when DDA mode is inactive")
    parser.add_argument("--delta", type=float, default=0.05, help="Step mutation variance scale for active DDA")

    args = parser.parse_args()
    
    run_visualization(
        algo=args.algo,
        env_type=args.env,
        model_path=args.model,
        use_dda=args.dda,
        fixed_speed=args.speed,
        delta=args.delta
    )