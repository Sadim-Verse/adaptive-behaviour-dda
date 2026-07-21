# scripts/run_ablations.py
"""
Run all ablation studies for Table 5.
- 4 ablation configurations: no_dda, fixed_difficulty, no_rolling_window, no_reward_smoothing
- 3 environments: single, multi, obstacle_minigrid (Procedural)
- 5 seeds: 42, 105, 777, 999, 2026
- 1 algorithm: DDQN (the primary agent, matches Table 5's own title)
- 2 modes: steady (always), spike (optional, --spike flag)

Default (steady only, all 3 envs): 4 x 3 x 5 = 60 runs
With --spike: 4 x 3 x 5 x 2 = 120 runs

Results saved to: ./logs/ablations/<ablation_name>/
Folder naming (matches build_run_folder(), and matches what
analyze_ablations.py actually reads):
    {env}_ddqn_delta0p05_seed{seed}_abl_{ablation_name}          (steady)
    {env}_ddqn_delta0p05_seed{seed}_abl_{ablation_name}_spike    (spike)

WHY THIS VERSION EXISTS -- THE CLIP BUG REACHES FURTHER THAN "no_dda":
envs/chase_env.py and envs/multi_chase_env.py's set_difficulty() clipped to
6.0 regardless of what --max_speed was actually passed in. That means
fixed_difficulty, no_rolling_window, and no_reward_smoothing -- which all
explicitly request --max_speed 9.0 below -- were ALSO silently forced back
to 6.0 under the old (pre-fix) environment code, not just no_dda. All four
conditions need re-running under the corrected clip, not just the one that
"obviously" touches difficulty.

CHALLENGE_SPEED (no_dda) IS NOW PER-ENVIRONMENT, NOT ONE SHARED CONSTANT:
single/multi use 6.0 (a physical speed, ~67% of their real 9.0 ceiling).
obstacle_minigrid uses 0.6 (~60% of its real 1.0 ceiling, on its own
normalised [0,1] scale) -- chosen to match single/multi's relative
challenge level rather than either reusing the physical-speed number
verbatim (which would silently clip to 1.0 = absolute max, a DIFFERENT and
harder condition than intended) or defaulting to the scale's max outright.

BUG FIXED IN THIS VERSION (the actual reason this file changed again):
The ABLATIONS dict's "no_dda" entry has cmd_args=None, meant to be resolved
per-environment inside run_ablation() -- but that resolution was never
actually wired in; run_ablation() still unconditionally did
cmd.extend(ablation_config["cmd_args"]), which is a TypeError the moment
cmd_args is None (list.extend(None) fails immediately). Fixed below: the
no_dda case is now handled explicitly before the generic extend.

Spike-mode runs are OFF by default (--spike to enable). Ablation latency is
N/A for any condition that hasn't had a spike-mode run yet.
"""

import os
import sys
import subprocess
import time
import argparse

# ── Configuration ─────────────────────────────────────────────────────────── #

SEEDS = [42, 105, 777, 999, 2026]
ENVIRONMENTS = ["single", "multi", "obstacle_minigrid"]
SPIKE_EPISODE = 250
BASE_LOG_DIR = "./logs/ablations"

# Challenge speed for "No DDA" ablation, per environment -- see module
# docstring for why this can't be one shared constant (single/multi use a
# physical speed scale with ceiling 9.0; obstacle_minigrid uses a
# normalised [0,1] scale with ceiling 1.0 -- reusing 6.0 there would
# silently clip to 1.0, a different and harder condition than intended).
CHALLENGE_SPEED = {
    "single": 6.0,
    "multi": 6.0,
    "obstacle_minigrid": 0.6,
}

# Path to trained DDQN model per environment. Single/Multi keep the extra
# per-algo subfolder here deliberately -- this is the MODEL CHECKPOINT path
# (best_model.zip), a separate convention from where EVALUATION OUTPUT gets
# saved (confirmed flat, no subfolder, via statistical_tests.py and
# analyze_ablations.py's corrected path resolution). Do not "fix" this to
# match the flat output structure -- they are legitimately different real
# folders.
MODEL_PATH = {
    "single": "./logs/experiments/ddqn/single_ddqn_delta0p05_seed42/best_model.zip",
    "multi": "./logs/experiments/ddqn/multi_ddqn_delta0p05_seed42/best_model.zip",
    "obstacle_minigrid": "./logs/procedural_env_v2/obstacle_minigrid_ddqn_delta0p05_seed42/best_model.zip",
}

# ── Ablation Configurations ──────────────────────────────────────────────── #

ABLATIONS = {
    "no_dda": {
        "description": "No DDA Controller -- fixed speed at challenge level (per-environment, see CHALLENGE_SPEED)",
        "cmd_args": None,  # resolved per-environment in run_ablation(), NOT here -- see bug note above
        "ablation_flag": "no_dda",
    },
    "fixed_difficulty": {
        "description": "DDA Controller exists but holds speed constant",
        "cmd_args": ["--max_speed", "9.0"],
        "ablation_flag": "fixed_difficulty",
    },
    "no_rolling_window": {
        "description": "DDA uses immediate reward instead of rolling average",
        "cmd_args": ["--max_speed", "9.0"],
        "ablation_flag": "no_rolling_window",
    },
    "no_reward_smoothing": {
        "description": "DDA uses raw rewards without any smoothing",
        "cmd_args": ["--max_speed", "9.0"],
        "ablation_flag": "no_smoothing",
    },
}

# ── Helper ────────────────────────────────────────────────────────────────── #

def build_run_folder(ablation_name, env, seed, spike=False):
    """
    Folder name as actually created by train_multi_env.py's main(), and
    actually read by analyze_ablations.py. Pattern:
        {env}_ddqn_delta0p05_seed{seed}[_spike]_abl_{ablation}
    (spike suffix comes BEFORE the ablation suffix -- matches
    train_multi_env.py's own run_id construction order.)
    """
    spike_part = "_spike" if spike else ""
    return f"{env}_ddqn_delta0p05_seed{seed}{spike_part}_abl_{ablation_name}"


def run_ablation(ablation_name, env, seed, spike=False):
    model_path = MODEL_PATH[env]
    run_folder = build_run_folder(ablation_name, env, seed, spike)
    ablation_config = ABLATIONS[ablation_name]

    cmd = [
        sys.executable, "-m", "training.train_multi_env",
        "--algo", "ddqn",
        "--env", env,
        "--resume", model_path,
        "--delta", "0.05",
        "--seed", str(seed),
        "--output", os.path.join(BASE_LOG_DIR, ablation_name),
    ]

    # no_dda's --max_speed is resolved per-environment (CHALLENGE_SPEED),
    # NOT a fixed cmd_args list -- this is the actual bug fix in this
    # version, see module docstring.
    if ablation_name == "no_dda":
        cmd.extend(["--max_speed", str(CHALLENGE_SPEED[env])])
    else:
        cmd.extend(ablation_config["cmd_args"])

    if ablation_config["ablation_flag"] is not None:
        cmd.extend(["--ablation", ablation_config["ablation_flag"]])
    if spike:
        cmd.extend(["--spike", str(SPIKE_EPISODE)])

    print(f"  Running: {' '.join(cmd)}")
    print(f"  Saving to: {run_folder}")

    start_time = time.time()
    try:
        subprocess.run(cmd, check=True, capture_output=False)
    except subprocess.CalledProcessError as e:
        print(f"  FAILED: {e}")
        return False
    print(f"  Completed in {time.time() - start_time:.1f}s")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--envs", nargs="+", default=ENVIRONMENTS, choices=ENVIRONMENTS)
    ap.add_argument("--ablations", nargs="+", default=list(ABLATIONS.keys()), choices=list(ABLATIONS.keys()))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--spike", action="store_true",
                     help="Also run spike-mode variants (needed for Table 5's "
                          "Adaptation Latency column). Roughly doubles runtime "
                          "for whatever subset of --ablations/--envs is selected.")
    args = ap.parse_args()

    modes = [False, True] if args.spike else [False]
    total = len(args.ablations) * len(args.envs) * len(args.seeds) * len(modes)

    print("=" * 80)
    print("ABLATION STUDIES")
    print(f"  Ablations: {', '.join(args.ablations)}")
    print(f"  Environments: {args.envs}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Modes: {'steady + spike' if args.spike else 'steady only'}")
    print(f"  Challenge Speed (No DDA), per environment: {CHALLENGE_SPEED}")
    print(f"  Total runs: {total}")
    print("=" * 80)

    failed = 0
    done = 0

    for ablation_name in args.ablations:
        cfg = ABLATIONS[ablation_name]
        print(f"\n{'-'*80}")
        print(f"ABLATION: {ablation_name.upper()}  ({cfg['description']})")
        print(f"{'-'*80}")

        for env in args.envs:
            print(f"\n  Environment: {env.upper()}")
            for seed in args.seeds:
                for spike in modes:
                    done += 1
                    tag = "spike" if spike else "steady"
                    print(f"\n    [{done}/{total}] Seed {seed} ({tag})...")
                    if not run_ablation(ablation_name, env, seed, spike):
                        failed += 1

    print("\n" + "=" * 80)
    print("Ablation studies complete.")
    print(f"  Total runs: {total}")
    print(f"  Failed: {failed}")
    print(f"  Results saved to: {BASE_LOG_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()