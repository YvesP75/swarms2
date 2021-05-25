import numpy as np
from dataclasses import dataclass

import param_
from drone import Drone
from dronemodel import DroneModel
from settings import Settings


@dataclass
class Team:
    """
    Creates a team (it is either red or blue / foe or friend
    """

    is_blue: bool
    drones: [Drone]
    drone_model: DroneModel

    def reset(self, obs=None):
        if obs:
            for drone, obs in zip(self.drones, obs):
                drone.reset(obs=obs)
        else:
            for drone in self.drones:
                drone.reset()

    def get_observation(self) -> np.ndarray:
        """
        get the observation for the RL agent
        :return: observation in the form of flatten np.arrays of shape(squad_number, 6*squad_size)
        """
        obs = np.array([drone.get_observation() for drone in self.drones])
        deads = ~np.array([drone.is_alive for drone in self.drones])

        return obs, deads

    def step(self, action: np.ndarray):
        obs = np.zeros((len(self.drones), 6))
        done = np.zeros((len(self.drones),))
        reward = np.zeros((len(self.drones),))
        for index, drone in enumerate(self.drones):
            obs[index], reward[index], done[index], _ = drone.step(action[index])
        done = (sum(done) == len(self.drones))
        return obs, sum(reward), done, 0


class BlueTeam(Team):
    """
    Creates the blue team
    """

    def __init__(self, number_of_drones: int = Settings.blues):
        self.is_blue = True
        self.drone_model = DroneModel(self.is_blue)

        # initialise blue positions and speeds
        positions = np.zeros((number_of_drones, 3))
        speeds = np.zeros((number_of_drones, 3))
        blue_speed = Settings.blue_speed_init * self.drone_model.max_speed
        circle = index = 0
        for d in range(number_of_drones):
            positions[d] = np.array([min(Settings.blue_circles_rho[circle] * Settings.blue_distance_factor,
                                         param_.PERIMETER),
                                     Settings.blue_circles_theta[circle] + index * 2 * np.pi / 3,
                                     Settings.blue_circles_zed[circle]])
            clockwise = 1 - 2 * (circle % 2)
            speeds[d] = np.array([blue_speed, np.pi / 6 * clockwise, 0])
            index += 1
            if index == Settings.blues_per_circle[circle]:
                index = 0
                circle += 1

        self.drones = [Drone(is_blue=True, position=position, speed=speed, id_=id_)
                       for (id_, position, speed) in zip(range(number_of_drones), positions, speeds)]


class RedTeam(Team):
    """
    Creates the red team
    """

    def __init__(self, number_of_drones: int = Settings.reds):
        self.is_blue = False
        self.drone_model = DroneModel(self.is_blue)

        positions = np.zeros((number_of_drones, 3))
        speeds = np.zeros((number_of_drones, 3))
        speed_rho = Settings.red_speed_init * self.drone_model.max_speed
        squad = index = 0
        for d in range(number_of_drones):
            positions[d] = [min(Settings.red_squads_rho[squad] * Settings.red_distance_factor
                            + np.random.rand() * Settings.red_rho_noise[squad],
                                param_.PERIMETER),
                            Settings.red_squads_theta[squad] + np.random.rand() * Settings.red_theta_noise[squad],
                            Settings.red_squads_zed[squad] + np.random.rand() * Settings.red_zed_noise[squad]]
            speeds[d] = [speed_rho, np.pi + positions[d][1], 0]
            index += 1
            if index == Settings.red_squads[squad]:
                index = 0
                squad += 1

        self.drones = [Drone(is_blue=False, position=position, speed=speed, id_=id_)
                       for (id_, position, speed) in zip(range(len(positions)), positions, speeds)]
