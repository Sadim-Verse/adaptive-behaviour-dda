# scripts/check_ppo_distribution.py
"""
Check PPO reward distribution.
"""

import numpy as np
import matplotlib.pyplot as plt

# Load data
rewards = np.load("./logs/ablations/no_dda/single_ppo_delta0p05_seed42_abl_no_dda/episode_rewards.npy")

print("=" * 60)
print("PPO NO DDA REWARD DISTRIBUTION")
print("=" * 60)
print(f"Number of episodes: {len(rewards)}")
print(f"Reward range: {np.min(rewards):.1f} to {np.max(rewards):.1f}")
print(f"Mean: {np.mean(rewards):.1f}")
print(f"Std: {np.std(rewards):.1f}")

# Count episodes in each zone
below_600 = np.sum(rewards < 600)
between_600_1200 = np.sum((rewards >= 600) & (rewards <= 1200))
above_1200 = np.sum(rewards > 1200)

print(f"\nEpisodes below 600: {below_600} ({below_600/len(rewards)*100:.1f}%)")
print(f"Episodes in [600, 1200]: {between_600_1200} ({between_600_1200/len(rewards)*100:.1f}%)")
print(f"Episodes above 1200: {above_1200} ({above_1200/len(rewards)*100:.1f}%)")

# Plot histogram
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(rewards, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(600, color='red', linestyle='--', label='Flow Low (600)')
ax.axvline(1200, color='blue', linestyle='--', label='Flow High (1200)')
ax.set_xlabel('Episode Reward')
ax.set_ylabel('Frequency')
ax.set_title('PPO No DDA Reward Distribution')
ax.legend()
plt.savefig('./logs/figures/ppo_no_dda_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Histogram saved to ./logs/figures/ppo_no_dda_distribution.png")