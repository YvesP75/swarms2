from dataclasses import dataclass
from scipy.integrate import odeint
import numpy as np

import param_


@dataclass
class DroneModel:
    """
    Creates a drone_model of a drone
    """

    def __init__(self, is_blue):
        self.drone_model = param_.DRONE_MODELS[param_.DRONE_MODEL[is_blue]]

        self.angle_to_neutralisation = self.drone_model['angle_to_neutralisation']
        self.distance_to_neutralisation = self.drone_model['distance_to_neutralisation']
        self.duration_to_neutralisation = self.drone_model['duration_to_neutralisation']

        self.Cxy = self.drone_model['Cxy']
        self.Cz = self.drone_model['Cz']
        self.mass = self.drone_model['mass']

        self.Fxy_ratio = self.drone_model['Fxy_ratio']
        self.Fz_min_ratio = self.drone_model['Fz_min_ratio']
        self.Fz_max_ratio = self.drone_model['Fz_max_ratio']

        self.weight_eq = self.mass * param_.g * (1 - self.Fz_min_ratio)
        self.Fz_plus = (self.Fz_max_ratio - 1) * self.mass * param_.g
        self.Fz_minus = (1 - self.Fz_min_ratio) * self.mass * param_.g
        self.Fxy = self.mass * param_.g * self.Fxy_ratio

        self.max_speed = np.sqrt(self.Fxy / self.Cxy)
        self.max_up_speed = np.sqrt(self.Fz_plus / self.Cz)
        self.max_down_speed = np.sqrt(self.Fz_minus / self.Cz)
        self.max_rot_speed = 2 * np.pi

    def get_trajectory(self, pos_xyz, speed_xyz, action: np.ndarray(3,), time_: np.ndarray(1,)) -> np.ndarray(3,):
        '''
        returns next position given the current position, speed and applied forces
        :param pos_xyz:
        :param speed_xyz:
        :param action:
        :param time_:
        :return:
        '''

        rho = action[0]  # in 0, 1
        theta = 2*np.pi * action[1]  # in 0, 2pi
        psy = np.pi * (action[2] - 0.5)  # in -pi/2, pi/2

        fx = rho * np.cos(theta) * np.cos(psy) * self.Fxy
        fy = rho * np.sin(theta) * np.cos(psy) * self.Fxy
        fz = rho * np.sin(psy) * (self.Fz_plus if 0 < psy else self.Fz_minus)

        pos_speed = np.hstack((pos_xyz, speed_xyz))

        result_ = odeint(
            lambda u, v: self.drone_dynamics(u, v, fx, fy, fz, self.Cxy, self.Cz, self.mass),
            pos_speed,
            time_,
            Dfun=lambda u, v: self.fulljac(u, v, self.Cxy, self.Cz, self.mass)
        )
        x, y, z, dx, dy, dz = result_.T

        return np.array([x, y, z], dtype='float32'), np.array([dx, dy, dz], dtype='float32')

    def drone_dynamics(self, pos_speed, time_, f_x, f_y, f_z, Cxy, Cz, m):
        x, y, z, dx, dy, dz = pos_speed
        return [dx,
                dy,
                dz,
                1/m * (f_x - Cxy * dx * np.sqrt(dx**2 + dy**2 + dz**2)),
                1/m * (f_y - Cxy * dy * np.sqrt(dx**2 + dy**2 + dz**2)),
                1/m * (f_z - Cz * dz * np.sqrt(dx**2 + dy**2 + dz**2))]

    def fulljac(self, pos_speed, time_, Cxy, Cz, m) -> np.ndarray((6, 6), ):
        '''
        returns the Jacobian of the differential equation of the trajectory
        :param pos_speed:
        :param time_:
        :param Cxy:
        :param Cz:
        :param m:
        :return:
        '''

        x, y, z, dx, dy, dz = pos_speed
        J = np.zeros((6, 6))
        J[0, 3] = 1
        J[1, 4] = 1
        J[2, 5] = 1
        J[3, 3] = -Cxy/m * ((np.sqrt(dx**2 + dy**2 + dz**2)) + dx**2 / np.sqrt(dx**2 + dy**2 + dz**2))
        J[3, 4] = -Cxy/m * (dx * dy / np.sqrt(dx**2 + dy**2 + dz**2))
        J[3, 5] = -Cxy/m * (dx * dz / np.sqrt(dx**2 + dy**2 + dz**2))
        J[4, 4] = -Cxy/m * ((np.sqrt(dx**2 + dy**2 + dz**2)) + dy**2 / np.sqrt(dx**2 + dy**2 + dz**2))
        J[4, 3] = -Cxy/m * (dy * dx / np.sqrt(dx**2 + dy**2 + dz**2))
        J[4, 5] = -Cxy/m * (dy * dz / np.sqrt(dx**2 + dy**2 + dz**2))
        J[5, 5] = -Cz/m * ((np.sqrt(dx**2 + dy**2 + dz**2)) + dz**2 / np.sqrt(dx**2 + dy**2 + dz**2))
        J[5, 3] = -Cz/m * (dz * dx / np.sqrt(dx**2 + dy**2 + dz**2))
        J[5, 4] = -Cz/m * (dz * dy / np.sqrt(dx**2 + dy**2 + dz**2))
        return J
