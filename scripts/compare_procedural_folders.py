import numpy as np
import os

SEEDS = [42, 105, 777, 999, 2026]
PATHS = {
    "dsens (26.6)": "./logs/delta_sensitivity_spike/obstacle_minigrid_ddqn_delta0p05_seed{seed}_spike",
    "penv2 (75.4)": "./logs/procedural_env_v2/obstacle_minigrid_ddqn_delta0p05_seed{seed}_spike",
}

print(f"{'seed':>6} | {'dsens latency':>14} | {'penv2 latency':>14} | {'dsens prespike peak':>20} | {'penv2 prespike peak':>20}")
print("-" * 90)

for seed in SEEDS:
    row = []
    for label, template in PATHS.items():
        path = template.format(seed=seed)
        lat_file = os.path.join(path, "adaptation_latency.npy")
        roll_file = os.path.join(path, "rolling_avg.npy")
        if not os.path.exists(lat_file):
            row.append(("MISSING", "MISSING"))
            continue
        lat = float(np.load(lat_file)[0])
        rolling = np.load(roll_file)
        peak = float(np.max(rolling[:250])) if len(rolling) >= 250 else float("nan")
        row.append((lat, peak))

    (dsens_lat, dsens_peak), (penv2_lat, penv2_peak) = row
    print(f"{seed:>6} | {str(dsens_lat):>14} | {str(penv2_lat):>14} | {str(dsens_peak):>20} | {str(penv2_peak):>20}")