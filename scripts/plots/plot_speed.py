# scripts/plots/plot_speed.py
"""
Generate four images of speed trajectories:
1. Single, Steady
2. Single, Spike
3. Multi, Steady
4. Multi, Spike
Each image shows all 6 delta values in a 2×3 grid.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ── CREATE OUTPUT DIRECTORY ──
os.makedirs('./logs/figures', exist_ok=True)

# ── CONFIGURATION ──
deltas = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]

# Define the four configurations
configs = [
    {
        "name": "single_steady",
        "label": "Single Environment, Steady Mode",
        "base_dir": "./logs/delta_sensitivity",
        "suffix": "",
        "color": "#2c3e50"
    },
    {
        "name": "single_spike",
        "label": "Single Environment, Spike Mode",
        "base_dir": "./logs/delta_sensitivity_spike",
        "suffix": "_spike",
        "color": "#e74c3c"
    },
    {
        "name": "multi_steady",
        "label": "Multi Environment, Steady Mode",
        "base_dir": "./logs/delta_sensitivity",
        "suffix": "",
        "color": "#2c3e50"
    },
    {
        "name": "multi_spike",
        "label": "Multi Environment, Spike Mode",
        "base_dir": "./logs/delta_sensitivity_spike",
        "suffix": "_spike",
        "color": "#e74c3c"
    }
]

# ── GENERATE ALL FOUR IMAGES ──
for config in configs:
    # Determine environment prefix
    env_prefix = "single" if "single" in config["name"] else "multi"
    
    # Load data for all deltas
    speed_data = []
    missing = []

    for delta in deltas:
        delta_str = str(delta).replace('.', 'p')
        file_path = f"{config['base_dir']}/{env_prefix}_ddqn_delta{delta_str}_seed42{config['suffix']}/speed_history.npy"
        
        if os.path.exists(file_path):
            speed = np.load(file_path)
            speed_data.append(speed)
        else:
            missing.append(delta)
            speed_data.append(None)

    if missing:
        print(f"⚠️ {config['label']}: Missing data for deltas {missing}")

    # ── CREATE FIGURE ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Difficulty Oscillation Dynamics — {config["label"]}', 
                 fontsize=16, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        if i < len(speed_data) and speed_data[i] is not None:
            speed = speed_data[i]
            episodes = np.arange(len(speed))
            ax.plot(episodes, speed, linewidth=1.5, color=config["color"])
            
            # Min/max speed lines
            ax.axhline(y=2.0, color='green', linestyle='--', alpha=0.5, linewidth=0.8)
            ax.axhline(y=9.0, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
            
            # Spike line (only for spike modes)
            if "spike" in config["name"]:
                ax.axvline(x=250, color='orange', linestyle='--', alpha=0.6, linewidth=0.8)
            
            ax.set_title(f'Δ = {deltas[i]:.2f}', fontsize=12)
            ax.set_xlabel('Episode', fontsize=10)
            ax.set_ylabel('Speed', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 500)
            ax.set_ylim(0, 10)
        else:
            ax.set_title(f'Δ = {deltas[i]:.2f} (Missing)', fontsize=12)
            ax.text(0.5, 0.5, 'Data not found', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, 500)
            ax.set_ylim(0, 10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    output_name = f"figure8_speed_{config['name']}_all_deltas.png"
    plt.savefig(f'./logs/figures/{output_name}', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_name}")

print("\n✅ All four images saved to ./logs/figures/")
print("  - figure8_speed_single_steady_all_deltas.png")
print("  - figure8_speed_single_spike_all_deltas.png")
print("  - figure8_speed_multi_steady_all_deltas.png")
print("  - figure8_speed_multi_spike_all_deltas.png")