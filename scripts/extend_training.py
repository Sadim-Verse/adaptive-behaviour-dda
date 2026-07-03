# scripts/extend_training.py
import os
from stable_baselines3 import DQN
from training.train_multi_env import make_env

# Configuration paths
MODEL_PATH = r"logs\experiments\multi_ddqn_delta0p05_seed42\best_model.zip"
EXTENDED_LOG_DIR = r"logs\experiments\multi_ddqn_extended"
os.makedirs(EXTENDED_LOG_DIR, exist_ok=True)

print("============ RESUMING DDQN TRAINING PROFILE ============")
# 1. Recreate the exact MultiChaseEnv environment
env = make_env(env_type="multi", initial_speed=2.0)

# 2. Load the trained model weights and bind them to the environment
print(f"Loading existing checkpoint weights from: {MODEL_PATH}")
model = DQN.load(MODEL_PATH, env=env)

# 3. Define the additional block of timesteps (e.g., 2,000,000 more steps)
ADDITIONAL_TIMESTEPS = 3_000_000 
print(f"Executing extended training loop for {ADDITIONAL_TIMESTEPS} additional steps...")

# 4. Kick off training without resetting the internal timestep counter
model.learn(
    total_timesteps=ADDITIONAL_TIMESTEPS,
    reset_num_timesteps=False,  # CRITICAL: Keeps the timeline continuous in TensorBoard
    tb_log_name="multi_ddqn_extended"
)

# 5. Save the extended variations
final_path = os.path.join(EXTENDED_LOG_DIR, "extended_final_model.zip")
model.save(final_path)
print(f"✅ Success! Extended model saved cleanly to: {final_path}")
print("========================================================")