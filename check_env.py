from stable_baselines3.common.env_checker import check_env
from gym_wrapper import AdaptiveGameEnv

# Initialize environment
env = AdaptiveGameEnv()

# Run the official check suite 
print("Checking environment compatibility...")
try:
    check_env(env)
    print("Environment check passed! Your Gym is compliant.")
except Exception as e:
    print(f"Environment check failed: {e}")

# Quick Random Test (Does it run without crashing?)
print("\nRunning a quick random simulation...")
obs, _ = env.reset()
for i in range(10):
    # Pick a random action (0, 1, or 2)
    action = env.action_space.sample()

    # Step the environment 
    obs, reward, terminated, _, _ = env.step(action)

    print(f"Step {i+1}: Action={action}, Reward={reward}, Obs={obs}")

    if terminated:
        print("Agent died! Resetting...")
        env.reset()

print("Simulation complete.")