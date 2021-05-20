from dataclasses import dataclass

import numpy as np

from drone import Drone


@dataclass
class SwarmPolicy:
    blues: int
    reds: int
    is_blue: bool

    def predict(self, obs):

        return self._simple_predict(obs)

    def _simple_predict(self, obs):
        blue_obs, red_obs, _, _ = obs
        simple_obs = blue_obs if self.is_blue else red_obs
        drone = Drone(is_blue=self.is_blue)
        action = []
        for d_obs in simple_obs:
            pos_n, speed_n = d_obs[:3], d_obs[3:6]
            pos = drone.from_norm(pos_n, drone.max_positions, drone.min_positions)
            drone.position = pos
            speed = drone.from_norm(speed_n, drone.max_speeds, drone.min_speeds)
            drone.speed = speed
            action_d = drone.simple_red()
            action.append(action_d)

        del drone  # the drone was created for positions manipulations

        return np.array(action)
