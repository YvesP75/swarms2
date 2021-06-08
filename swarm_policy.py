from dataclasses import dataclass
import numpy as np
from stable_baselines3 import SAC
from os import path

import param_
from drone import Drone


@dataclass
class SwarmPolicy:
    blues: int
    reds: int
    is_blue: bool
    model: object = None
    count: int = 0

    def __post_init__(self):

        dir_path = "policies/last" + f"/b{self.blues}r{self.reds}/"
        model_path = dir_path + ("blues_last.zip" if self.is_blue else "reds_last.zip")
        if path.exists(model_path):
            print("model loaded:" + model_path)
            self.model = SAC.load(model_path, verbose=1)

    # predicts from the model or from an simple centripete model
    def predict(self, obs):

        self.count += 1

        if self.model:
            action, _ = self.model.predict(obs)
            # verbose = 'prediction from ' + (' blue model' if self.is_blue else ' red model') + ' at ' + str(self.count)
            # print(verbose)
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

        action = _centralise(action)
        return action


def _decentralise(obs):  # [-1,1] to [0,1]
    obs = (obs+1)/2
    return obs


def _centralise(act):  # [0,1] to [-1,1]
    act = (act - 1/2) * 2
    return act


def calculate_target(blue_drone: Drone, red_drone: Drone) -> np.ndarray(3, ):
    '''

    :param blue_drone:
    :param red_drone:
    :return:
    '''

    def transform(pos, delta_, theta_):
        pos[0] -= delta_
        pos[1] -= theta_
        return pos[0] * np.exp(1j * pos[1])

    def untransform_to_array(pos, delta_, theta_):
        pos[0] += delta_
        pos[1] += theta_
        return pos

    theta = red_drone.position[1]
    delta = param_.GROUNDZONE

    z_blue = transform(blue_drone.position, delta, theta)
    z_red = np.real(transform(red_drone.position, delta, theta))

    v_blue = blue_drone.drone_model.max_speed
    v_red = red_drone.drone_model.max_speed

    blue_shooting_distance = blue_drone.drone_model.distance_to_neutralisation

    blue_time_to_zero = (np.abs(z_blue) - blue_shooting_distance) / v_blue
    red_time_to_zero = z_red / v_red

    if red_time_to_zero <= param_.STEP or red_time_to_zero < blue_time_to_zero + param_.STEP:
        return np.zeros(3), red_time_to_zero
    else:
        max_target = z_red
        min_target = 0
        while True:
            target = (max_target + min_target) / 2
            blue_time_to_target = max(0, (np.abs(z_blue - target) - blue_shooting_distance) / v_blue)
            red_time_to_target = np.abs(z_red - target) / v_red
            if red_time_to_target - param_.STEP < blue_time_to_target <= red_time_to_target:
                target = untransform_to_array((target / z_red) * red_drone.position, delta, theta)
                return target, blue_time_to_target
            if red_time_to_target < blue_time_to_target:
                max_target = target
                min_target = min_target
            else:  # blue_  time_to_target  <= red_time_to_target -1:
                max_target = max_target
                min_target = target




def unitary_test(rho_blue: float, theta_blue: float, rho_red: float, theta_red: float):
    '''
    tests for the calculate target function
    :param rho_blue:
    :param theta_blue:
    :param rho_red:
    :param theta_red:
    :return:
    '''
    blue_drone = Drone()
    blue_drone.position = np.array([rho_blue, theta_blue, 100])
    red_drone = Drone(is_blue=False)
    red_drone.position = np.array([rho_red, theta_red, 100])
    tg, time = calculate_target(blue_drone, red_drone)
    print('rho_blue : ', rho_blue, ' theta_blue : ', theta_blue, ' rho_red : ', rho_red, ' theta_red : ', theta_red,
          ' tg : ', tg, ' time : ', time)
    return tg, time


def test():
    '''
    test for the calculate trajectory function
    :return:
    '''
    for rho_blue in [1000]:
        for theta_blue in np.pi * np.array([-1, 0.75, 0.5, 0.25, 0]):
            for rho_red in [1000]:
                for theta_red in np.pi * np.array([0, 1/4]):
                    unitary_test(rho_blue=rho_blue, theta_blue=theta_blue, rho_red=rho_red, theta_red=theta_red)
    print('done')



