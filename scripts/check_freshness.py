import os
import glob

# Paste PowerShell's Get-Item output VERBATIM, no reformatting needed --
# already filled in from your last message:
CLIP_FIX_TIME = "Saturday, July 11, 2026 10:40:19 PM"        # later of chase_env.py / multi_chase_env.py
DETECTION_FIX_TIME = "Saturday, July 18, 2026 6:43:20 AM"    # train_multi_env.py

import datetime

def parse_time(s):
    # Strip the leading day-of-week ("Saturday, ") if present, then parse
    # PowerShell's "July 11, 2026 10:40:19 PM" format directly -- no need
    # to hand-reformat anything pasted from Get-Item's default output.
    s = s.strip()
    if "," in s and s.split(",")[0].strip().isalpha():
        s = s.split(",", 1)[1].strip()
    return datetime.datetime.strptime(s, "%B %d, %Y %I:%M:%S %p")

if CLIP_FIX_TIME is None or DETECTION_FIX_TIME is None:
    print("Fill in CLIP_FIX_TIME and DETECTION_FIX_TIME at the top of this script first.")
    print("Get them with:")
    print(r"  (Get-Item .\envs\chase_env.py).LastWriteTime")
    print(r"  (Get-Item .\envs\multi_chase_env.py).LastWriteTime")
    print(r"  (Get-Item .\training\train_multi_env.py).LastWriteTime")
    raise SystemExit(1)

clip_fix = parse_time(CLIP_FIX_TIME)
detection_fix = parse_time(DETECTION_FIX_TIME)

print(f"Clip fix cutoff (single/multi only):      {clip_fix}")
print(f"Detection fix cutoff (all environments):   {detection_fix}")
print()

stale_clip = []
stale_detection = []
missing = []
ok = []

for mode in ["steady", "spike"]:
    base = f"./logs/delta_sensitivity_{mode}"
    if not os.path.isdir(base):
        print(f"MISSING base dir: {base}")
        continue
    for folder in sorted(os.listdir(base)):
        path = os.path.join(base, folder)
        rewards_file = os.path.join(path, "episode_rewards.npy")
        if not os.path.exists(rewards_file):
            continue
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(rewards_file))

        env = folder.split("_ddqn_")[0]
        needs_clip_check = env in ("single", "multi")

        if needs_clip_check and mtime < clip_fix:
            stale_clip.append((mode, folder, mtime))
        elif mtime < detection_fix:
            stale_detection.append((mode, folder, mtime))
        else:
            ok.append((mode, folder, mtime))

print(f"OK (post both relevant fixes): {len(ok)}")
print(f"STALE -- predates clip fix (single/multi): {len(stale_clip)}")
for m, f, t in stale_clip:
    print(f"    [{m}] {f}  (mtime {t})")
print(f"STALE -- predates detection fix (any env): {len(stale_detection)}")
for m, f, t in stale_detection:
    print(f"    [{m}] {f}  (mtime {t})")

if stale_clip or stale_detection:
    print("\nRecommendation: delete the stale folders listed above (not the whole tree),")
    print("then re-run run_delta_sensitivity.py for both modes -- the cache-skip logic")
    print("will regenerate only what's missing, leaving already-fresh folders untouched.")
else:
    print("\nAll existing folders are fresh relative to both fixes. Safe to run")
    print("run_delta_sensitivity.py directly -- it will fill in any genuinely missing")
    print("combinations without needing to delete anything first.")