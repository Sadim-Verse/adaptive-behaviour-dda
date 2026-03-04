import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import os

# Import your custom env
from gym_wrapper import AdaptiveGameEnv

# The Shock Callback 
class DifficultyShockCallback(BaseCallback):
    """
    This callback triggers the 'Shock Event' at a specific timestep.
    It simulates the environment changing unexpectedly.
    """
    def __init__(self,shock_step: int, new_difficulty: int, verbose=0):
        super(DifficultyShockCallback, self).__init__(verbose)
        self.shock_step = shock_step
        self.new_difficulty = new_difficulty
        self.shock_triggered = False

    def _on_step(self) -> bool:
        # Check if we hit the specific step count
        if not self.shock_triggered and self.num_timesteps >= self.shock_step:
            print(f"\n⚡ SHOCK EVENT TRIGGERED at step {self.num_timesteps} ⚡")
            print(f"   -> Changing Enemy Speed to {self.new_difficulty}")
            
            # We use env_method because SB3 wraps the environment in a 'VecEnv'
            # This calls 'set_difficulty(new_difficulty)' inside your wrapper
            self.training_env.env_method("set_difficulty", self.new_difficulty)
            
            self.shock_triggered = True
            
        return True
    
# Setup Directrories
# Create folders to save models and logs
models_dir = "models/DQN"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Initialize the Environment
# Wrap it in 'Monitor' to log data for TensorBoard (Graphing)
env = AdaptiveGameEnv(render_mode=None) # No render = Faster training
env = Monitor(env, log_dir)

# Define the AI Model (The Brain)
# Use DQN (Deep Q Network) with a Multi-Layer Perceptron (MlpPolicy)
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=0.0003,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32
)

# Configuration
TOTAL_TIMESTEPS = 1000000
SHOCK_STEP = 500000          # Change difficulty halfway through
NEW_DIFFICULTY = 7         # Hard mode (Speed = 15)

# Initialize the callback
shock_callback = DifficultyShockCallback(shock_step=SHOCK_STEP, new_difficulty=NEW_DIFFICULTY)

# Start Training
print("Training Started...")
print(f"    -> Phase 1: Easy Mode (0 - {SHOCK_STEP} steps)")
print(f"    -> Phase 2: Hard Mode ({SHOCK_STEP}+ steps)")

model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=shock_callback)

print("Training Complete!")

# Save the Model
model.save(f"{models_dir}/adaptive_agent_final")
print("Model Saved.")