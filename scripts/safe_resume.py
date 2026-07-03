# scripts/bulletproof_resume.py
import os
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from training.train_multi_env import make_env

# 1. Point this to the HIGHEST valid model you can find in your folders
# (Either your 3M step 'best_model.zip' or a higher checkpoint if found)
BASE_MODEL_PATH = r"logs\experiments\multi_ddqn_delta0p05_seed42\best_model.zip"

# 2. Set up a dedicated, safe directory for this recovery run
RECOVERY_DIR = r"logs\experiments\multi_ddqn_recovery"
os.makedirs(RECOVERY_DIR, exist_ok=True)

def main():
    print("=" * 65)
    print("🛡️ INITIALIZING BULLETPROOF RECOVERY TRAINING")
    print(f"  Source Model: {BASE_MODEL_PATH}")
    print(f"  Saving Logs To: {RECOVERY_DIR}")
    print("=" * 65)

    # Recreate environments safely
    env = make_env(env_type="multi", initial_speed=2.0)
    vec_env = DummyVecEnv([lambda: env])
    eval_env = DummyVecEnv([lambda: make_env(env_type="multi", initial_speed=2.0)])

    # Load the model and bind it to the training environment
    print("Loading model weights into memory...")
    model = DQN.load(BASE_MODEL_PATH, env=vec_env)

    # Define robust callbacks to save to disk frequently
    checkpoint_cb = CheckpointCallback(
        save_freq=100_000,       # Saves a hard backup snapshot every 100k steps
        save_path=RECOVERY_DIR,
        name_prefix="recovery_checkpoint"
    )
    
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=RECOVERY_DIR, # Overwrites with the absolute best policy seen
        log_path=RECOVERY_DIR,
        eval_freq=50_000,        # Evaluates performance every 50k steps
        n_eval_episodes=10,
        deterministic=True,
        verbose=1
    )

    # Set how many total steps you want to run (e.g., 3,000,000 more steps)
    ADDITIONAL_STEPS = 3_000_000 
    print(f"\n🚀 Starting training loop for an additional {ADDITIONAL_STEPS:,} steps.")
    print("Progress will auto-save to disk periodically. If your laptop dies, your data is safe!")
    
    model.learn(
        total_timesteps=ADDITIONAL_STEPS,
        reset_num_timesteps=False, # Keeps your TensorBoard charts continuous
        callback=[checkpoint_cb, eval_cb],
        tb_log_name="multi_ddqn_recovery"
    )

    # Save final model state
    final_path = os.path.join(RECOVERY_DIR, "recovery_final_model")
    model.save(final_path)
    print(f"\n✅ Training complete! Final model saved to: {final_path}.zip")

if __name__ == "__main__":
    main()