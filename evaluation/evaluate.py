import numpy as np
from stable_baselines3 import DQN

def measure_adaptation_latency(model_path, env_creator, dda_controller, 
                               shock_episode=100, recovery_threshold=0.8):
    """
    Simulate sudden difficulty spike and measure recovery steps.
    """
    model = DQN.load(model_path)
    env = env_creator()
    
    # Run episodes normally until shock
    episode_rewards = []
    for ep in range(shock_episode):
        obs, _ = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        episode_rewards.append(ep_reward)
        dda_controller.update(ep_reward)
        env.set_difficulty(dda_controller.current_speed)
    
    # Record pre-shock performance peak
    pre_shock_peak = max(episode_rewards[-10:])  # Peak of last 10 episodes
    target_reward = pre_shock_peak * recovery_threshold
    
    # Apply shock: manually set difficulty to max
    env.set_difficulty(2.0)
    shock_time = env.current_step  # approximate
    
    # Continue running and measure when rolling avg recovers
    recovery_ep = None
    while recovery_ep is None and len(episode_rewards) < 500:
        obs, _ = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        episode_rewards.append(ep_reward)
        dda_controller.update(ep_reward)
        env.set_difficulty(dda_controller.current_speed)
        
        # Check if rolling average has recovered
        if len(dda_controller.reward_buffer) >= 10:
            rolling_avg = np.mean(list(dda_controller.reward_buffer))
            if rolling_avg >= target_reward:
                recovery_ep = len(episode_rewards)
                break
    
    if recovery_ep is not None:
        adaptation_latency_episodes = recovery_ep - shock_episode
        print(f"Adaptation Latency: {adaptation_latency_episodes} episodes")
        return adaptation_latency_episodes
    else:
        print("Agent did not recover within episode limit")
        return None