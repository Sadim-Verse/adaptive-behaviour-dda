"""
scripts/generate_figure3_reward_stability.py
================================================
Figure 3: Reward Stability.

Requirements from supervisor feedback:
  - Flow Zone shading
  - spike annotation
  - recovery threshold line
  - difficulty trajectory overlay

Generates 4 figures: {multi, obstacle_minigrid} x {steady, spike}.
Steady-mode figures omit the spike marker and recovery threshold line
(nothing to annotate -- no perturbation occurs in steady mode); they
still show Flow Zone shading and the difficulty trajectory overlay.

Uses the MAIN evaluation battery's default delta=0.05, seed=42 run for
each environment/mode -- the standard representative configuration used
throughout this project's other tables and figures, not a delta-
sensitivity variant.

Usage:
    python -m scripts.generate_figure3_reward_stability
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from training.train_multi_env import FLOW_ZONES

BASE_LOG_DIRS = {
    "single": "./logs/experiments",
    "obstacle_minigrid": "./logs/procedural_env_v2",
}
ENV_TITLES = {"single": "Single", "obstacle_minigrid": "Procedural"}
SPIKE_EPISODE = 250
RECOVERY_FRACTION = 0.80  # matches run_dda_only()'s recovery_threshold = pre_spike_peak * 0.80


def run_dir(env, mode):
    suffix = "_spike" if mode == "spike" else ""
    folder = f"{env}_ddqn_delta0p05_seed42{suffix}"
    return os.path.join(BASE_LOG_DIRS[env], folder)


def load_run(env, mode):
    d = run_dir(env, mode)
    rewards = np.load(os.path.join(d, "episode_rewards.npy"))
    rolling_avg = np.load(os.path.join(d, "rolling_avg.npy"))
    speed = np.load(os.path.join(d, "speed_history.npy"))
    latency = None
    lat_path = os.path.join(d, "adaptation_latency.npy")
    if mode == "spike" and os.path.exists(lat_path):
        latency = float(np.asarray(np.load(lat_path)).flatten()[0])
    return rewards, rolling_avg, speed, latency


def generate_figure(env, mode, output_dir):
    rewards, rolling_avg, speed, latency = load_run(env, mode)
    flow_low, flow_high = FLOW_ZONES[env]["default"]
    episodes = np.arange(1, len(rewards) + 1)

    fig, ax1 = plt.subplots(figsize=(11, 5.5))

    # Flow Zone shading
    ax1.axhspan(flow_low, flow_high, color="#2ecc71", alpha=0.15,
                label=f"Flow Zone [{flow_low:.1f}, {flow_high:.1f}]")
    ax1.axhline(flow_low, color="#27ae60", linewidth=0.8, linestyle=":")
    ax1.axhline(flow_high, color="#27ae60", linewidth=0.8, linestyle=":")

    # reward trajectory (raw, faint) + rolling average (bold, what the
    # controller actually acts on)
    ax1.plot(episodes, rewards, color="#95a5a6", alpha=0.35, linewidth=0.7, label="Episode reward (raw)")
    ax1.plot(episodes, rolling_avg, color="#2980b9", linewidth=1.6, label="Rolling avg (10-ep window)")

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward", color="#2980b9")
    ax1.tick_params(axis="y", labelcolor="#2980b9")

    # difficulty trajectory overlay on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(episodes, speed, color="#e67e22", linewidth=1.3, alpha=0.85, label="Difficulty (speed)")
    ax2.set_ylabel("Difficulty (DDA speed)", color="#e67e22")
    ax2.tick_params(axis="y", labelcolor="#e67e22")

    title_suffix = ""
    if mode == "spike":
        # spike annotation
        ax1.axvline(SPIKE_EPISODE, color="#c0392b", linewidth=1.4, linestyle="--")
        ax1.annotate("SPIKE\n(forced to max difficulty)",
                     xy=(SPIKE_EPISODE, ax1.get_ylim()[1] * 0.92),
                     fontsize=8.5, color="#c0392b", ha="left",
                     bbox=dict(boxstyle="round,pad=0.3", fc="#fdecea", ec="#c0392b"))

        # recovery threshold line: 80% of the pre-spike peak rolling avg
        pre_spike_window = rolling_avg[max(0, SPIKE_EPISODE - 10):SPIKE_EPISODE]
        if len(pre_spike_window) > 0:
            pre_spike_peak = float(np.max(pre_spike_window))
            recovery_threshold = pre_spike_peak * RECOVERY_FRACTION
            ax1.axhline(recovery_threshold, color="#8e44ad", linewidth=1.2, linestyle="-.",
                        label=f"Recovery threshold ({RECOVERY_FRACTION:.0%} of pre-spike peak)")

        if latency is not None:
            title_suffix = f"  |  Adaptation latency: {latency:.0f} episodes"

    ax1.set_title(f"{ENV_TITLES[env]} — Reward Stability ({mode.capitalize()} Mode){title_suffix}",
                  fontsize=12, fontweight="bold")

    # combined legend from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    fname = f"figure3_reward_stability_{env}_{mode}.png"
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}")


def main(output_dir="./report_figures"):
    os.makedirs(output_dir, exist_ok=True)
    for env in ["single", "obstacle_minigrid"]:
        for mode in ["steady", "spike"]:
            print(f"Generating {env}/{mode}...")
            try:
                generate_figure(env, mode, output_dir)
            except FileNotFoundError as e:
                print(f"  MISSING DATA: {e}")


if __name__ == "__main__":
    main()