import numpy as np
import os

# These three numbers are not guesses -- each was independently confirmed
# at least twice already in this project, via completely separate scripts
# reading completely separate folders:
#   Single:      statistical_tests.py (./logs/experiments) AND
#                analyze_ablations.py Full System row
#   Multi:       same two sources
#   Procedural:  analyze_ablations.py Full System row, after the earlier
#                investigation into the 26.6-vs-75.4 discrepancy concluded
#                75.4 was the correct, detection-fix-applied value
CONFIRMED_CORRECT = {
    "single":            (86.6, 33.0),
    "multi":              (145.2, 27.4),
    "obstacle_minigrid":   (75.4, 64.7),
}

N_EPISODES = 500

for env, (expected_mean, expected_std) in CONFIRMED_CORRECT.items():
    seeds = [42, 105, 777, 999, 2026]
    latencies = []
    for seed in seeds:
        path = f"./logs/delta_sensitivity_spike/{env}_ddqn_delta0p05_seed{seed}_spike/adaptation_latency.npy"
        if not os.path.exists(path):
            print(f"{env}: MISSING {path}")
            continue
        latencies.append(float(np.load(path)[0]))

    latencies = np.array(latencies)
    recovered = latencies[latencies < N_EPISODES]
    mean = np.mean(recovered) if len(recovered) else float("nan")
    std = np.std(recovered, ddof=1) if len(recovered) > 1 else 0.0

    match = abs(mean - expected_mean) < 0.5 and abs(std - expected_std) < 0.5
    verdict = "MATCH -- this env's delta_sensitivity_spike data is fine, ignore the timestamp warning" \
              if match else \
              "MISMATCH -- this env's delta_sensitivity_spike data is genuinely stale, needs deleting + re-running"

    print(f"{env:20} | computed: {mean:.1f} +/- {std:.1f}  |  confirmed-correct: {expected_mean} +/- {expected_std}  |  {verdict}")