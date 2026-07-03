# scripts/verify_baseline.py
"""
Deterministic baseline validation script.
Evaluates a trained model under STATIC difficulty conditions (No active DDA).
Directly populates "Without DDA" / "Fixed Difficulty" metrics for Table 5.
"""
import os
import argparse
import numpy as np
from stable_baselines3 import DQN, PPO
from training.train_multi_env import make_env

def verify_static_model(algo, env_type, model_path, speed=2.0, n_episodes=5):
    print("=" * 65)
    print(f"STATIC GROUND-TRUTH VERIFICATION (NO DDA)")
    print(f"  Algorithm   : {algo.upper()}")
    print(f"  Environment : {env_type.upper()}")
    print(f"  Fixed Speed : {speed}")
    print("=" * 65)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not locate model file at: {model_path}")

    # Instantiate environment with fixed speed baseline
    env = make_env(env_type=env_type, initial_speed=speed)
    
    # Load model class safely based on training architecture
    if algo.lower() in ("dqn", "ddqn"):
        model = DQN.load(model_path)
    elif algo.lower() == "ppo":
        model = PPO.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm sequence '{algo}'.")
        
    episode_rewards = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=100 + ep)  # Standardize evaluation initialization
        total_reward = 0.0
        
        # Max steps capped at 2000 per episode via TimeLimit wrapper
        for step in range(2000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
                
        episode_rewards.append(total_reward)
        print(f"  ➡️ Episode {ep + 1:2d} Complete | Accumulated Reward: {total_reward:6.1f}")
        
    mean_rew = np.mean(episode_rewards)
    std_rew = np.std(episode_rewards)
    print("-" * 65)
    print(f"📊 SUMMARY OVER {n_episodes} STATIC EPISODES:")
    print(f"  Mean Performance Reward : {mean_rew:.1f} (±{std_rew:.1f})")
    if mean_rew >= 2000.0 if env_type == "single" else mean_rew >= 6000.0:
        print("  Status: Excellent convergence. Policy mastered this fixed difficulty baseline.")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify trained model under static conditions.")
    parser.add_argument("--algo", type=str, required=True, choices=["dqn", "ddqn", "ppo"])
    parser.add_argument("--env", type=str, required=True, choices=["single", "multi"])
    parser.add_argument("--model", type=str, required=True, help="Path to best_model.zip")
    parser.add_argument("--speed", type=float, default=2.0, help="Fixed velocity baseline")
    parser.add_argument("--episodes", type=int, default=5, help="Number of validation episodes")
    
    args = parser.parse_args()
    verify_static_model(args.algo, args.env, args.model, args.speed, args.episodes)
    