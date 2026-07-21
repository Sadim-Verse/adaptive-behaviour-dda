"""
ObstacleMiniGridEnv
--------------------
Procedurally-generated MiniGrid environment with moving, homing hazards.

Design goals (matching the existing ChaseEnv / MultiChaseEnv interface so the
DDA Controller, training script, and benchmarking pipeline need no changes):

- Discrete(3) action space: 0 = left, 1 = right, 2 = stay (same as ChaseEnv).
- set_difficulty(d) with d in [0, 1], called by the DDA Controller at episode
  boundaries. Drives grid size AND hazard density together:
    * grid width/height  (bigger grid -> more exposure time)
    * number of hazards  (more hazards -> more simultaneous threats)
    * hazard move interval (smaller interval -> hazards move more often,
      i.e. faster -> the discrete analogue of ChaseEnv's enemy_speed)
- Dense reward: +1 survival per step, -100 on collision, small distance bonus
  (mirrors MultiChaseEnv), so the existing Flow Zone / percentile machinery
  keeps working unmodified.
- Fixed-size flat observation vector (NOT MiniGrid's native image encoding).
  Because hazard COUNT varies with difficulty, the observation is padded to
  a fixed number of hazard "slots" so the observation_space shape never
  changes across difficulty levels (required by SB3).
- Procedural generation: grid layout (hazard start positions, agent start
  position) is freshly randomised every reset(), which is the actual
  Priority 1 fix -- previously each episode reused the same fixed layout
  logic, just with randomised initial conditions.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from minigrid.core.grid import Grid
from minigrid.core.world_object import WorldObj, Wall
from minigrid.core.mission import MissionSpace
from minigrid.core.constants import COLORS
from minigrid.minigrid_env import MiniGridEnv


class Hazard(WorldObj):
    """A moving, homing hazard tile. Movement logic lives in the env step(),
    not here -- WorldObj itself is just a rendered marker, same division of
    responsibility as enemy_x/enemy_y arrays in MultiChaseEnv."""

    def __init__(self):
        # Colored yellow (not red) so hazards are visually distinct from
        # the player's agent triangle, which MiniGrid renders in a
        # hardcoded red internally (Grid.render_tile) -- not something this
        # env controls, so the hazard color is what we can change instead.
        super().__init__("lava", "yellow")

    def can_overlap(self):
        # Agent can step onto a hazard cell -- that's how we detect collision
        return True

    def render(self, img):
        c = COLORS[self.color]
        # simple filled diamond so hazards are visually distinct from walls
        import numpy as _np
        mid = img.shape[0] // 2
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if abs(i - mid) + abs(j - mid) < mid:
                    img[i, j] = c


class ObstacleMiniGridEnv(MiniGridEnv):
    """
    Procedurally-generated hazard-avoidance environment.

    Difficulty parameter d in [0, 1] controls:
      grid size         : FIXED at INTERIOR_SIZE for d <= GRID_GROWTH_THRESHOLD,
                           then grows linearly up to INTERIOR_SIZE + GRID_GROWTH_EXTRA
                           at d=1.0 (see rationale below)
      num_hazards        : density(d) * (current interior area)
      vertical_speed      : continuous cells/step fall rate
      horizontal_speed    : continuous cells/step homing rate (kept < 1.0 so the
                            player always retains a per-step speed advantage)

    History of this design, briefly (see conversation for full diagnostics):
    v1 scaled grid size linearly across the WHOLE range alongside hazard
    count. That caused a non-monotonic survival curve under a strong
    lookahead policy (e.g. d=0.6 easier than d=0.4/0.5), because grid area
    grows quadratically while a linear hazard count doesn't keep pace --
    diluting density in some ranges. v2 fixed grid size entirely and drove
    difficulty only through hazard density + speed. That fixed the
    mid-range non-monotonicity, but empirical testing (both a
    privileged-information lookahead policy AND an actual trained PPO
    agent) showed d=1.0 had become a genuine ceiling: PPO's evaluated
    performance was IDENTICAL (9.33 survival) at every checkpoint from
    step 12,000 to 60,000 -- a classic signature of policy collapse to a
    constant action, which happens when the reward carries no learnable
    signal because outcome is nearly action-independent (death is close to
    unavoidable regardless of policy).

    v3 (this version) grows the grid ONLY above GRID_GROWTH_THRESHOLD, as a
    release valve for the ceiling specifically, while leaving the d <=
    threshold range exactly as validated in v2 (fixed grid, clean
    monotonicity, no directional conflict). This preserves the working
    range's validated behavior and targets the one region where we have
    direct evidence relief is needed.
    """

    INTERIOR_SIZE = 12          # fixed playable width/height for d <= GRID_GROWTH_THRESHOLD
    GRID_GROWTH_THRESHOLD = 0.7  # grid size only starts growing above this d
    GRID_GROWTH_EXTRA = 4        # interior grows up to INTERIOR_SIZE + this, at d=1.0

    # Hazard count is driven by DENSITY (hazards per unit area), not a raw
    # count linear in d. Reason: interior grid AREA grows quadratically
    # with d (area = (BASE_GRID + d*GRID_RANGE)^2), but a naive linear-in-d
    # hazard count only grows linearly -- so true threat density (and
    # therefore true difficulty under a policy that can exploit open space)
    # was NON-MONOTONIC. Confirmed with a greedy lookahead policy (picks
    # the action maximizing next-step min-distance to any hazard, i.e. a
    # much stronger player than the earlier naive nearest-hazard heuristic):
    # mean survival went 429 (d=0.0) -> 394 (d=0.2) -> 78 (d=0.4) -> 57
    # (d=0.5) -> 178 (d=0.6) -- d=0.6 was easier than both d=0.4 and d=0.5,
    # because hazard count had plateaued at 3 across that range while grid
    # area kept growing, diluting density. Tying count to density * area
    # fixes this at the source rather than re-tuning the count curve by hand.
    MIN_DENSITY = 1.0 / 64.0     # hazards per grid cell at d=0
    MAX_DENSITY = 3.0 / 64.0     # hazards per grid cell at d=1.
    # Validated with a greedy lookahead policy (80-200 seeds) against the
    # FIXED 12x12 interior: mean survival 217 (d=0.0) -> 272 (d=0.2, floor-
    # effect noise, see below) -> 83 (d=0.4) -> 36 (d=0.5) -> 22 (d=0.6) ->
    # 16 (d=0.8) -> 13 (d=1.0). Clean monotonic decline from d=0.2 onward.
    # One residual irregularity remains at d=0.0-0.2, where num_hazards is
    # only 1-2 -- at such small integers, individual episode outcomes are
    # dominated by where that single hazard happens to spawn/wander rather
    # than by systematic difficulty, so mean survival is noisy at the floor
    # even with 200 seeds (d=0.1 measured LOWER than both d=0.0 and d=0.2).
    # This is measurement variance at low counts, not the same structural
    # reversal the fixed-grid redesign was meant to fix, and it sits in the
    # least consequential part of the range for DDA testing (very-low
    # difficulty is supposed to be easy; the exact ordering among "easy"
    # settings matters far less than avoiding reversals in the working
    # range where the controller actually operates). Documented as a known
    # limitation rather than further hand-tuned -- see conversation history
    # for the reasoning on when to stop chasing this.
    MAX_HAZARD_SLOTS = 20        # fixed obs size -- generous upper bound, see _sync_difficulty

    # Continuous speed accumulators (cells per step), replacing the earlier
    # integer move_interval scheme. Diagnostic: isolating each parameter
    # change between d=0.75 and d=1.0 (20-seed dodge-heuristic survival)
    # showed the integer interval scheme was the dominant cause of a sharp
    # non-smooth drop at the top of the range (28.65 -> 22.95 steps from the
    # vertical interval 2->1 change alone, 28.65 -> 21.7 from horizontal
    # interval 3->2 alone) -- both larger than the hazard-count contribution
    # (18.75 -> 14.95). The root cause: 1/interval is a reciprocal, so the
    # smallest possible integer step (2->1) produces the LARGEST possible
    # relative speed jump (100%), regardless of where the tuning constants
    # are set. A continuous accumulator removes the discontinuity rather
    # than just relocating it.
    MIN_VERTICAL_SPEED = 0.25    # cells/step at d=0 (slowest fall)
    MAX_VERTICAL_SPEED = 1.5     # cells/step at d=1 (fastest fall)
    # Horizontal homing speed is kept strictly below 1.0 cell/step even at
    # d=1, so the player (who always moves exactly 1 cell/step when
    # dodging) retains a guaranteed per-step speed advantage in x -- this
    # preserves the "evasion must stay mathematically possible" invariant
    # from the earlier interval-based fix, without needing an artificial
    # integer floor (which was itself a source of the same rounding
    # problem).
    MIN_HORIZONTAL_SPEED = 0.15  # cells/step at d=0 (slowest homing)
    MAX_HORIZONTAL_SPEED = 0.7   # cells/step at d=1 (fastest homing, still < player's 1.0)
    # Hazards spawn randomly within the top SPAWN_ZONE_FRACTION of the
    # interior grid, never in the rows immediately above the player. This
    # keeps the vertical spawn row genuinely random (the Bug 1 fix) while
    # guaranteeing every hazard has to fall a meaningful distance before it
    # can reach the player -- without this bound, a hazard can spawn
    # adjacent to the player's row and cause an unavoidable "spawn kill"
    # (observed: collisions at step 1-4 mixed in with step 30-78 once hy
    # was randomised across the full grid height).
    SPAWN_ZONE_FRACTION = 0.5
    MAX_GRID = INTERIOR_SIZE + GRID_GROWTH_EXTRA + 2  # +2 for walls; largest possible grid (at d=1.0), used to size obs_space normalization bound

    def __init__(self, render_mode=None, difficulty=0.0, max_steps=2000,
                 randomize_difficulty=False):
        """
        Parameters
        ----------
        difficulty : float
            Fixed difficulty in [0,1], used every episode UNLESS
            randomize_difficulty=True. Also the one-time bootstrap value
            used to size the grid before MiniGrid's own __init__ runs
            (before self.np_random exists) even when randomize_difficulty
            is enabled -- see _sync_difficulty().
        randomize_difficulty : bool
            If True, EVERY reset() samples a fresh difficulty uniformly
            from [0, 1], ignoring set_difficulty()/the fixed `difficulty`
            argument (except for the one bootstrap call in __init__,
            before np_random exists). This is domain-randomized training,
            NOT evaluation-time DDA control.

            Why this exists: training was run with difficulty fixed at
            the DDA_SCALE floor (0.0) for the entire 6M-step budget (no
            domain randomization, no DDA active during training -- see
            train_multi_env.py). Evaluating that policy's own logged
            episodes showed a stark capability cliff: reasonable, varied
            performance at d=0.3, but UNIFORM near-instant death at d=0.7
            (7-30 steps every single episode, no partial competence in
            between) -- the signature of a policy that never encountered
            anything resembling that difficulty during training, not a
            policy that's merely weak there. More training steps at
            d=0.0 cannot fix this: the agent was never given a gradient
            of experience to generalize from.

            Setting randomize_difficulty=True during training (and
            leaving it False during evaluation, where set_difficulty()
            must remain under the DDA Controller's deterministic control)
            exposes the policy to the full difficulty spectrum during
            learning, which is what the DDA Controller will actually walk
            it through at evaluation time. This changes the training
            objective from "master d=0.0" to "behave reasonably across
            the whole range" -- a harder, but far more relevant, thing
            to have learned. See DDA_SCALE / make_env() in
            train_multi_env.py for how this flag is threaded through
            for training vs. evaluation runs.
        """
        self._difficulty = float(np.clip(difficulty, 0.0, 1.0))
        self._pending_difficulty = self._difficulty
        self.randomize_difficulty = randomize_difficulty
        self.width = self.INTERIOR_SIZE + 2   # +2 for boundary walls -- fixed, not difficulty-dependent
        self.height = self.INTERIOR_SIZE + 2
        self.num_hazards = 1
        self.vertical_speed = self.MIN_VERTICAL_SPEED
        self.horizontal_speed = self.MIN_HORIZONTAL_SPEED
        self._sync_difficulty()

        mission_space = MissionSpace(mission_func=lambda: "avoid the hazards")

        super().__init__(
            mission_space=mission_space,
            width=self.width,
            height=self.height,
            max_steps=max_steps,
            see_through_walls=True,
            render_mode=render_mode,
        )

        # Override MiniGrid's default turn/forward action space with the
        # ChaseEnv-style left/right/stay scheme.
        self.action_space = spaces.Discrete(3)

        # Fixed-size flat observation: [player_x_norm,
        #   (dx_norm, dy_norm) per hazard slot, padded with 1.0 = "no threat"]
        obs_dim = 1 + 2 * self.MAX_HAZARD_SLOTS
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.hazards = []          # list of [x, y] grid coords, active hazards

    # ------------------------------------------------------------------ #
    #  Difficulty interface (called by DDA Controller at episode boundary) #
    # ------------------------------------------------------------------ #

    def _stochastic_round(self, value):
        """Round to an integer probabilistically based on the fractional
        part, so the *expected* value across many episodes/seeds varies
        smoothly even though any single episode gets an integer count.
        Needed because num_hazards only has 5 possible integer values
        across the full d in [0,1] range -- plain round() makes each
        transition (e.g. 1->2 hazards, a 100% relative jump) land at a
        single sharp difficulty threshold instead of phasing in gradually.
        Falls back to plain round() before np_random exists (the one-time
        __init__ pre-pass, before the env is ever reset)."""
        if not hasattr(self, "np_random") or self.np_random is None:
            return round(value)
        floor_val = int(np.floor(value))
        frac = value - floor_val
        return floor_val + (1 if self.np_random.random() < frac else 0)

    def _sync_difficulty(self):
        # Domain-randomized training: draw a fresh difficulty every episode
        # instead of using whatever set_difficulty()/the constructor's fixed
        # `difficulty` last specified. Guarded on self.np_random existing
        # because _sync_difficulty() is also called once from __init__,
        # BEFORE MiniGridEnv.__init__() has set up self.np_random -- that
        # one bootstrap call always uses the fixed _pending_difficulty
        # (harmless either way, since it only sizes the grid before
        # gymnasium/minigrid internals exist; every REAL reset() afterward
        # goes through the randomized branch below).
        if self.randomize_difficulty and getattr(self, "np_random", None) is not None:
            d = float(self.np_random.uniform(0.0, 1.0))
        else:
            d = self._pending_difficulty
        # Grid size: fixed below GRID_GROWTH_THRESHOLD (validated range,
        # unchanged from v2), grows linearly above it as a release valve
        # for the high-difficulty ceiling (see class docstring).
        if d <= self.GRID_GROWTH_THRESHOLD:
            interior = self.INTERIOR_SIZE
        else:
            growth_frac = (d - self.GRID_GROWTH_THRESHOLD) / (1.0 - self.GRID_GROWTH_THRESHOLD)
            interior = round(self.INTERIOR_SIZE + growth_frac * self.GRID_GROWTH_EXTRA)
        self.width = interior + 2
        self.height = interior + 2
        area = interior * interior
        density = self.MIN_DENSITY + d * (self.MAX_DENSITY - self.MIN_DENSITY)
        target_count = density * area
        self.num_hazards = int(np.clip(self._stochastic_round(target_count), 1, self.MAX_HAZARD_SLOTS))
        self.vertical_speed = self.MIN_VERTICAL_SPEED + d * (self.MAX_VERTICAL_SPEED - self.MIN_VERTICAL_SPEED)
        self.horizontal_speed = self.MIN_HORIZONTAL_SPEED + d * (self.MAX_HORIZONTAL_SPEED - self.MIN_HORIZONTAL_SPEED)
        self._difficulty = d

    def set_difficulty(self, difficulty):
        """Called by the DDA Controller. Takes effect at the NEXT reset(),
        matching the episode-boundary update timing of set_difficulty() in
        ChaseEnv/MultiChaseEnv.

        Has NO effect if randomize_difficulty=True (training mode) -- that
        mode always draws a fresh random difficulty every episode instead.
        This is intentional: training-time domain randomization and
        evaluation-time DDA control are mutually exclusive uses of this
        environment, and DDACallback is never attached during training
        (see train_multi_env.py), so this situation shouldn't arise in
        practice -- documented here so it's not a silent surprise if it
        ever does."""
        self._pending_difficulty = float(np.clip(difficulty, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    #  Procedural generation                                              #
    # ------------------------------------------------------------------ #

    def _gen_grid(self, width, height):
        self._sync_difficulty()
        width, height = self.width, self.height

        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Player starts at bottom-centre interior row (mirrors ChaseEnv)
        self.player_pos = [width // 2, height - 2]
        self.agent_pos = tuple(self.player_pos)
        self.agent_dir = 3  # facing "up" purely for rendering; unused in step()

        # Procedurally place hazards: random interior x AND y, re-rolled
        # every reset(). Both axes are now genuinely stochastic -- the
        # previous version randomised hx but computed hy purely from the
        # hazard's index (hy = 1 + i % ...), which made the closest hazard's
        # starting row -- and therefore the time-to-collision -- effectively
        # fixed across every seed. Excluding the player's own row keeps a
        # hazard from ever spawning directly on top of the player.
        self.hazards = []
        for i in range(self.num_hazards):
            hx = self.np_random.integers(1, width - 1)
            spawn_ceiling = max(2, round((height - 2) * self.SPAWN_ZONE_FRACTION))
            hy = self.np_random.integers(1, spawn_ceiling)
            self.hazards.append([int(hx), int(hy)])
            # Place into the actual rendered grid -- without this, hazards
            # are tracked for collision math but never drawn on screen.
            self.grid.set(int(hx), int(hy), Hazard())

        self._v_progress = np.zeros(self.num_hazards)
        self._h_progress = np.zeros(self.num_hazards)
        self.mission = "avoid the hazards"

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return self._get_obs(), self._get_info()

    # ------------------------------------------------------------------ #
    #  Step                                                                #
    # ------------------------------------------------------------------ #

    def step(self, action):
        # --- player movement: left / right / stay ---
        if action == 0:
            self.player_pos[0] = max(1, self.player_pos[0] - 1)
        elif action == 1:
            self.player_pos[0] = min(self.width - 2, self.player_pos[0] + 1)
        # action 2: stay
        self.agent_pos = tuple(self.player_pos)

        # --- hazard movement: continuous fractional-speed accumulators ---
        # Replaces the earlier integer move_interval scheme (see class
        # docstring for why: reciprocal-integer intervals produce a
        # disproportionate jump at interval=1, confirmed empirically).
        # Each hazard accumulates vertical_speed / horizontal_speed every
        # step; whenever the accumulator crosses 1.0 it moves one cell and
        # the accumulator resets by -1.0, so a hazard can move 0, 1, or
        # (at high difficulty) 2 cells in a single step, matching whatever
        # the continuous speed value implies -- same principle as
        # ChaseEnv's `enemy_y += self.enemy_speed`, just quantised to whole
        # grid cells instead of continuous pixels.
        #
        # Collision is checked after EVERY single-cell move, not just once
        # at the end of the step. At speeds > 1 cell/step a hazard can
        # cross two rows in one step() call; checking only the final
        # position would let it tunnel straight through the player's row
        # without ever coinciding with it (e.g. hazard 1 row above player,
        # moves 2 cells, lands 1 row below -- collision missed entirely).
        collision = False
        for i, hz in enumerate(self.hazards):
            self._v_progress[i] += self.vertical_speed
            self._h_progress[i] += self.horizontal_speed

            while (self._v_progress[i] >= 1.0 or self._h_progress[i] >= 1.0) and not collision:
                old_x, old_y = hz[0], hz[1]

                if self._h_progress[i] >= 1.0:
                    self._h_progress[i] -= 1.0
                    dx = np.sign(self.player_pos[0] - hz[0])
                    hz[0] = int(np.clip(hz[0] + dx, 1, self.width - 2))

                if self._v_progress[i] >= 1.0:
                    self._v_progress[i] -= 1.0
                    hz[1] += 1
                    if hz[1] > self.height - 2:
                        hz[1] = 1
                        hz[0] = int(self.np_random.integers(1, self.width - 1))
                        self._h_progress[i] = 0.0  # avoid a stray extra shift right after respawn

                if (hz[0], hz[1]) != (old_x, old_y):
                    self.grid.set(old_x, old_y, None)
                    self.grid.set(hz[0], hz[1], Hazard())

                if hz[0] == self.player_pos[0] and hz[1] == self.player_pos[1]:
                    collision = True

        self.step_count += 1
        if collision:
            reward = -100.0
            terminated = True
        else:
            terminated = False
            reward = 1.0
            if self.hazards:
                dists = [abs(hz[0] - self.player_pos[0]) + abs(hz[1] - self.player_pos[1])
                          for hz in self.hazards]
                min_dist = min(dists)
                max_possible = self.width + self.height
                reward += (min_dist / max_possible) * 0.5

        truncated = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ------------------------------------------------------------------ #
    #  Observation                                                        #
    # ------------------------------------------------------------------ #

    def _get_obs(self):
        px_norm = (self.player_pos[0] / self.width) * 2 - 1

        rel = []
        sorted_hz = sorted(
            self.hazards,
            key=lambda hz: abs(hz[0] - self.player_pos[0]) + abs(hz[1] - self.player_pos[1]),
        )
        for i in range(self.MAX_HAZARD_SLOTS):
            if i < len(sorted_hz):
                hz = sorted_hz[i]
                dx_norm = np.clip((hz[0] - self.player_pos[0]) / self.MAX_GRID, -1.0, 1.0)
                dy_norm = np.clip((hz[1] - self.player_pos[1]) / self.MAX_GRID, -1.0, 1.0)
                rel.extend([dx_norm, dy_norm])
            else:
                # padding slot: "no threat" sentinel, far away
                rel.extend([1.0, 1.0])

        return np.array([px_norm] + rel, dtype=np.float32)

    def _get_info(self):
        return {
            "difficulty": self._difficulty,
            "num_hazards": self.num_hazards,
            "vertical_speed": round(self.vertical_speed, 3),
            "horizontal_speed": round(self.horizontal_speed, 3),
            "grid_size": self.width,
        }