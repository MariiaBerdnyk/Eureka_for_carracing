from typing import Tuple

from gym.envs.box2d.car_racing import *
from gymnasium.error import InvalidAction

from env.car_racing_base import CarRacing


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

        truncated, terminated, step_reward = self.reward(action, truncated, terminated)

        if self.render_mode == "human":
            self.render()
        return self.state, step_reward, terminated, truncated, {}

    def reward(self, action, truncated, terminated, **kwargs) -> Tuple[bool, bool, float]:

        if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count == len(self.track) or self.new_lap:
                # Truncation due to finishing lap
                # This should not be treated as a failure
                # but like a timeout
                truncated = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                terminated = True
                step_reward = -100

        return truncated, terminated, step_reward
