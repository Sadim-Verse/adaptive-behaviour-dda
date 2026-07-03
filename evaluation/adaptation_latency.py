def measure_adaptation_latency(episode_rewards, shock_episode, window_size=10, recovery_frac=0.8):
    """
    Compute adaptation latency after a deliberate difficulty shock.

    Parameters
    ----------
    episode_rewards : list of float
        Total reward per episode, in chronological order.
    shock_episode : int
        Index of the episode where the shock was applied (0‑based).
    window_size : int
        Window size for rolling average.
    recovery_frac : float
        Fraction of pre‑shock peak that must be recovered.

    Returns
    -------
    latency : int or None
        Number of episodes from shock_episode to recovery, or None if not recovered.
    """
    if shock_episode < window_size:
        raise ValueError("shock_episode must be >= window_size to compute pre‑shock peak")

    # Pre‑shock rolling average over the window just before the shock
    pre_shock_rewards = episode_rewards[shock_episode - window_size : shock_episode]
    pre_shock_peak = max(pre_shock_rewards)
    target = pre_shock_peak * recovery_frac

    # Compute rolling averages after the shock
    rolling_avgs = []
    for i in range(shock_episode, len(episode_rewards)):
        start = i - window_size + 1
        if start >= shock_episode:   # window fully after shock
            window = episode_rewards[start : i+1]
            rolling_avgs.append(np.mean(window))

    # Find first episode where rolling average meets or exceeds target
    for offset, ra in enumerate(rolling_avgs):
        if ra >= target:
            return offset + window_size   # offset + window_size episodes after shock
    return None