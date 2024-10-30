def compute_reward(self, action: Union[np.ndarray, int], truncated: bool, terminated: bool, **kwargs) -> Tuple[bool, bool, float]:
    """
    Reward function designed to encourage completing the track as quickly and accurately as possible.

    Args:
        action (Union[np.ndarray, int]): The action taken by the car.
        truncated (bool): Whether the episode has been truncated prematurely.
        terminated (bool): Whether the episode has ended due to a terminal state.

    Returns:
        Tuple[bool, bool, float]: A tuple containing whether the episode is truncated, terminated, and the reward value.
    """

    # Initialize the reward value
    reward = 0.0

    # Check if action leads to unfavorable states
    if (truncated or terminated):  
        # Penalty for unfavorable state with entropy regularization
        penalty_with_entropy = -0.1 * np.log(np.sum(action**2)) + 0.05
        reward += penalty_with_entropy 

    # Reward the agent for visiting tiles on the track and completing laps
    tile_visited_count = len(self.contactListener_keepref.env.track)
    reward += (self.tile_visited_count / tile_visited_count) * 10.0

    # Introduce intermediate rewards for partial successes, such as reaching checkpoints or traveling a certain distance without leaving the track
    # Calculate distances and add rewards accordingly

    # Update the reward so it reflects the change from previous step
    self.prev_reward = self.reward
    self.reward += 5 * ((1 - (truncated or terminated)) * math.sin(self.t / 1000.0) / math.e)

    return truncated, terminated, self.reward
