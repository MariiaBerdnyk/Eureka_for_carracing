from typing import *

import torch
from gym.envs.box2d.car_racing import *
from gymnasium.error import InvalidAction

from gymnasium.envs.box2d.car_racing import CarRacing


class CarRacingCustom(CarRacing):
    def __init__(self, render_mode: Optional[str] = None, verbose: bool = False, lap_complete_percent: float = 0.95,
                 domain_randomize: bool = False, continuous: bool = True):
        super().__init__(render_mode, verbose, lap_complete_percent, domain_randomize, continuous)

    def step(self, action: Union[np.ndarray, int]):
        assert self.car is not None
        if action is not None:
            if self.continuous:
                self.car.steer(-action[0])
                self.car.gas(action[1])
                self.car.brake(action[2])
            else:
                if not self.action_space.contains(action):
                    raise InvalidAction(
                        f"you passed the invalid action `{action}`. "
                        f"The supported action_space is `{self.action_space}`"
                    )
                self.car.steer(-0.6 * (action == 1) + 0.6 * (action == 2))
                self.car.gas(0.2 * (action == 3))
                self.car.brake(0.8 * (action == 4))

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        self.state = self._render("state_pixels")

        truncated = False
        terminated = False
        step_reward = 0

        if action is not None:
            if self.tile_visited_count == len(self.track) or self.new_lap:
                # Truncation due to finishing lap
                # This should not be treated as a failure
                # but like a timeout
                truncated = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                terminated = True
                step_reward = -100
            truncated, terminated, step_reward = self.get_reward(action, truncated, terminated, step_reward)

        if self.render_mode == "human":
            self.render()
        return self.state, step_reward, terminated, truncated, {}

    def get_reward(self, action, truncated, terminated, step_reward) -> Tuple[bool, bool, float]:
        return compute_reward(self, action, truncated, terminated, step_reward)

def compute_reward(
        self,
        action: Union[np.ndarray, int],
        truncated: bool,
        terminated: bool,
        step_reward: float = 0,
) -> Tuple[bool, bool, float]:
    """Computes reward for CarRacing environment."""

    # Check if the car has finished a lap
    if self.tile_visited_count < len(self.track):
        step_reward += 10.0 / len(self.track)

    x, y = self.car.hull.position
    if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
        terminated = True
        step_reward -= 1.0

    # Introduce a bonus for reaching certain checkpoints along the track.
    if self.tile_visited_count % (len(self.track) // 5) == 0:
        step_reward += 20.0 / len(self.track)

    # If the car finishes the lap, we give a big reward to encourage it to finish the next lap as fast as possible.
    if self.tile_visited_count == len(self.track):
        truncated = True
        # scale rewards for better episodes
        if np.sum(self.rewards) > 100:
            step_reward *= 1.5

    return truncated, terminated, step_reward
