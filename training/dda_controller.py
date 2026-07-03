import numpy as np
from collections import deque


class DDAController:
    """
    Rule-based Dynamic Difficulty Adjustment using rolling average reward.

    Adjusts enemy speed after every episode according to a three-branch
    proportional feedback rule:

        If rolling_avg > flow_high  →  speed += delta  (too easy)
        If rolling_avg < flow_low   →  speed -= delta  (too hard)
        Otherwise                   →  no change       (in Flow Zone)

    Tracks rolling_avg_history so that Flow Zone Consistency can be
    computed correctly on the smoothed signal rather than raw episode rewards.
    """

    def __init__(self, window_size=10, flow_low=600.0, flow_high=1200.0,
                 delta=0.05, min_speed=2.0, max_speed=6.0, initial_speed=2.0):
        self.window_size  = window_size
        self.flow_low     = flow_low
        self.flow_high    = flow_high
        self.delta        = delta
        self.min_speed    = min_speed
        self.max_speed    = max_speed
        self.current_speed = initial_speed

        self.reward_buffer      = deque(maxlen=window_size)
        self.episode_rewards    = []          # raw per-episode rewards
        self.speed_history      = []          # speed after each episode
        self.rolling_avg_history = []         # rolling avg after each episode
        self.adjustment_history = []          # detailed per-episode record

    # ------------------------------------------------------------------ #

    def update(self, episode_reward):
        """
        Call once per episode with the total episode reward.
        Updates internal state and adjusts difficulty.

        Returns
        -------
        float : updated enemy speed
        """
        self.episode_rewards.append(episode_reward)
        self.reward_buffer.append(episode_reward)

        # Cannot compute a valid rolling average until the window is full
        if len(self.reward_buffer) < self.window_size:
            self.speed_history.append(self.current_speed)
            self.rolling_avg_history.append(episode_reward)  # use raw until window full
            return self.current_speed

        rolling_avg = float(np.mean(self.reward_buffer))
        self.rolling_avg_history.append(rolling_avg)

        old_speed = self.current_speed

        if rolling_avg > self.flow_high:
            self.current_speed = min(self.max_speed, self.current_speed + self.delta)
        elif rolling_avg < self.flow_low:
            self.current_speed = max(self.min_speed, self.current_speed - self.delta)

        self.speed_history.append(self.current_speed)
        self.adjustment_history.append({
            "episode":     len(self.episode_rewards),
            "rolling_avg": rolling_avg,
            "old_speed":   old_speed,
            "new_speed":   self.current_speed,
        })

        return self.current_speed

    # ------------------------------------------------------------------ #

    def get_flow_zone_consistency(self):
        """
        Compute Flow Zone Consistency over the full rolling_avg_history.

        FZC = proportion of episodes (after window warm-up) where the
        10-episode rolling average reward falls within [flow_low, flow_high].

        Returns
        -------
        float : FZC in [0, 1]
        """
        if len(self.rolling_avg_history) < self.window_size:
            return 0.0

        # Only count episodes after the window is full
        avgs = np.array(self.rolling_avg_history[self.window_size - 1:])
        in_zone = np.sum((avgs >= self.flow_low) & (avgs <= self.flow_high))
        return float(in_zone) / len(avgs) if len(avgs) > 0 else 0.0

    # ------------------------------------------------------------------ #

    def reset(self):
        """Reset all history. Useful when reusing the controller across runs."""
        self.current_speed       = self.min_speed
        self.reward_buffer       = deque(maxlen=self.window_size)
        self.episode_rewards     = []
        self.speed_history       = []
        self.rolling_avg_history = []
        self.adjustment_history  = []
