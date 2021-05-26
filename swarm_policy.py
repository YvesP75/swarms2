from dataclasses import dataclass
import numpy as np
from stable_baselines3 import SAC
from os import path

from drone import Drone
from settings import Settings


@dataclass
class SwarmPolicy:
    blues: int
    reds: int
    is_blue: bool
    model: object = None

    def __post_init__(self):

        dir_path = "policies/last" + f"/b{self.blues}r{self.reds}/"
        model_path = dir_path + ("blues_last.zip" if self.is_blue else "reds_last.zip")
        if path.exists(model_path):
            print(model_path)
            self.model = SAC.load(model_path, verbose=1)

    # predicts from the model or from an simple centripete model
    def predict(self, obs):

        if self.model:
            action, _ = self.model.predict(obs)
            return action
        else:
            return self._simple_predict(obs)

    # the default policy
    def _simple_predict(self, obs):
        simple_obs = _decentralise(obs[0:self.blues*6] if self.is_blue else obs[self.blues*6:(self.blues+self.reds)*6])
        drone = Drone(is_blue=self.is_blue)
        action = np.array([])
        nb_drones = self.blues if self.is_blue else self.reds
        for d in range(nb_drones):
            pos_n, speed_n = simple_obs[d*6:d*6+3], simple_obs[d*6+3:d*6+6]
            pos = drone.from_norm(pos_n, drone.max_positions, drone.min_positions)
            drone.position = pos
            speed = drone.from_norm(speed_n, drone.max_speeds, drone.min_speeds)
            drone.speed = speed
            action_d = drone.simple_red()
            action = np.hstack((action, action_d))

        del drone  # the drone was created for positions manipulations

        return np.array(action)


def _decentralise(obs):  # [-1,1] to [0,1]
    obs = (obs+1)/2
    return obs
