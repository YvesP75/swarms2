from dataclasses import dataclass
import numpy as np
from dronemodel import DroneModel

import param_ as param_
from settings import Settings


@dataclass
class Drone:
    """
    Creates a drone (it is either red or blue / foe or friend
    """
    is_blue: bool = True
    position: np.ndarray((3,)) = np.zeros((3,))
    drone_model: DroneModel = None
    max_speeds: np.ndarray((3,)) = None
    min_speeds: np.ndarray((3,)) = None
    init_position: np.ndarray((3,)) = None
    init_speed: np.ndarray((3,)) = None
    color = np.ndarray((3,))
    is_alive: bool = True
    is_fired: int = 0
    fires = 0
    step_ = 0
    id_: int = -1
    ttl: float = param_.DURATION  # ttl = Time To Live expressed in seconds
    speed: np.ndarray((3,)) = np.zeros((3,))
    min_positions = np.zeros((3,))
    max_positions = np.array([Settings.perimeter, 2*np.pi, Settings.perimeter_z])

    def __post_init__(self):
        self.drone_model = DroneModel(self.is_blue)
        self.max_speeds = np.array([self.drone_model.max_speed,
                                    2*np.pi,
                                    self.drone_model.max_up_speed])
        self.min_speeds = np.array([0,
                                    0,
                                    -self.drone_model.max_down_speed])

        self.init_position = self.position
        self.init_speed = self.speed
        self.color = param_.BLUE_COLOR if self.is_blue else param_.RED_COLOR
        self.ttl = self.position[0] / self.max_speeds[0] * param_.TTL_RATIO

    def reset(self):
        self.is_alive = True
        self.position = self.init_position
        self.speed = self.init_speed
        self.color = param_.BLUE_COLOR if self.is_blue else param_.RED_COLOR

    def step(self, action):
        self.step_ = self.step_ + 1  # for debug purposes
        reward = 0
        info = {'ttl': param_.DURATION}
        if self.is_alive:  # if the drone is dead, it no longer moves :)
            pos_xyz, speed_xyz = self.to_xyz(self.position), self.to_xyz(self.speed)
            pos_s, speed_s = \
                self.drone_model.get_trajectory(pos_xyz, speed_xyz, action, np.linspace(0, param_.STEP, 10))
            pos, speed = pos_s.T[-1], speed_s.T[-1]
            self.position, self.speed = self.from_xyz(pos), self.from_xyz(speed)
            self.ttl -= param_.STEP
            info['ttl'] = self.ttl
                
            if self._out_of_bounds():
                coef = -1 if self.is_blue else 1
                reward = coef * param_.OOB_COST
                self.is_alive = False
                info['oob'] = 1
                # if self.is_blue:
                #    print("another blue is oob")
                # else:
                #   print("another red is oob")
            else:
                if self._hits_target():
                    info['hits_target'] = 1
                    reward = -param_.TARGET_HIT_COST
                #    print("another red hits the target")
                    self.color = param_.RED_SUCCESS_COLOR
                    self.is_alive = False  # the red has done its job ...

        obs = self.get_observation()
        done = not self.is_alive

        return obs, reward, done, info

    def _out_of_bounds(self):
        return not (0 < self.position[2] < Settings.perimeter_z and self.position[1] < Settings.perimeter)

    def _hits_target(self):
        if self.is_blue:
            return False
        else:
            return np.sqrt(self.position[0]**2 + self.position[2]**2) < Settings.groundzone

    def fires_(self, foe) -> bool:
        """
        checks if the foe drone is hit by self
        :param foe: a foe drone
        :return: True= yes, got you
        """
        # deads don't kill nor die
        if not (self.is_alive and foe.is_alive):
            return False

        # lets see if foe is in the "fire cone"
        pos_xyz = - self.to_xyz(self.position) + self.to_xyz(foe.position)
        distance = np.linalg.norm(pos_xyz)
        pos_xyz /= distance

        if distance < self.drone_model.distance_to_neutralisation:
            vit_xyz = self.to_xyz(self.speed)
            vit_xyz /= np.linalg.norm(vit_xyz)
            cos_theta = np.dot(pos_xyz, vit_xyz)
            if 0 < cos_theta:
                theta = np.arccos(cos_theta)
                it_is_a_hit = theta < self.drone_model.angle_to_neutralisation
                return it_is_a_hit

        return False

    # tell the drones that they are dead
    def is_killed(self, is_blue=True):
        self.is_alive = False
        self.position[2] = 0
        self.color = param_.BLUE_DEAD_COLOR if is_blue else param_.RED_DEAD_COLOR

    def to_xyz(self, rho_theta_z: np.ndarray(shape=(3,))) -> np.ndarray(shape=(3,)):
        """
        allows to get the 3D xyz coordinates from a polar representation
        :param rho_theta_z: array (3,) with rho in meter, theta in rad, zed in meter for positions, /s for speeds, etc.
        :return: float array (3,) with x, y, z in meter, /s for speeds, etc.
        """
        xy_ = rho_theta_z[0] * np.exp(1j * rho_theta_z[1])
        return np.array([np.real(xy_), np.imag(xy_), rho_theta_z[2]])

    def from_xyz(self, xyz: np.ndarray(shape=(3,))) -> np.ndarray(shape=(3,)):
        """
        """
        z_complex = xyz[0] + 1j*xyz[1]
        rho = np.abs(z_complex)
        theta = np.angle(z_complex)
        return np.array([rho, theta, xyz[2]], dtype='float32')

    def to_norm(self,
                rho_theta_z: np.ndarray(shape=(3,)),
                max_vector: np.ndarray(shape=(3,)),
                min_vector: np.ndarray(shape=(3,)) = np.array([0, 0, 0]))\
            -> np.ndarray(shape=(3,), dtype='float32'):
        """
        normalises the position/speed in order to have all space in a [0;1]**3 space
        :return: rho, theta, zed in a [0;1]**3 space
        """
        rho = rho_theta_z[0] / max_vector[0]
        theta = (rho_theta_z[1] / (2 * np.pi)) % 1
        zed = (rho_theta_z[2] - min_vector[2]) / (max_vector[2] - min_vector[2])
        return np.array([rho, theta, zed], dtype='float32')

    def from_norm(self,
                  norm: np.ndarray(shape=(3,)),
                  max_vector: np.ndarray(shape=(3,)),
                  min_vector: np.ndarray(shape=(3,)) = np.array([0, 0, 0]))\
            -> np.ndarray(shape=(3,), dtype='float32'):
        """
        denormalises and renders into cylindric coordinates
        :param norm:
        :param max_vector:
        :param min_vector:
        :return:
        """
        rho = norm[0] * max_vector[0]
        theta = norm[1] * 2*np.pi
        zed = norm[2] * (max_vector[2] - min_vector[2]) + min_vector[2]
        return np.array([rho, theta, zed], dtype='float32')

    def to_lat_lon_zed(self, lat, lon):
        z = self.position[0] * np.exp(1j * self.position[1])
        lat = np.imag(z) * 360 / (40075 * 1000) + lat
        lon = np.real(z) * 360 / (40075 * 1000 * np.cos(np.pi / 180 * lat)) + lon
        return lat, lon, self.position[2]

    def distance(self, other_drone=None):

        if other_drone:
            distance = np.sqrt(np.abs(self.position[0] * np.exp(1j * self.position[1]) -
                                      other_drone.position[0] * np.exp(1j * other_drone.position[1])) ** 2 +
                               (self.position[2] - other_drone.position[2]) ** 2)
        else:
            distance = np.sqrt((self.position[0] ** 2) + self.position[2] ** 2)
        return distance

    def get_observation(self):  # -> np.array(shape=(6,), dtype='float32'):
        """
        get normalised and transformed position and speed
        :return:
        """
        # calculates transformed normalised position
        normalised_position = self.to_norm(self.position, self.max_positions, self.min_positions)

        # calculates transformed normalised speed
        normalised_speed = self.to_norm(self.speed, self.max_speeds, self.min_speeds)

        return np.append(normalised_position, normalised_speed)

    def simple_red(self) -> np.ndarray(shape=(3,)):

        theta = (self.position[1] + np.pi) / (2*np.pi) % 1

        # slope of drone given its position
        tan_phi = np.sign(self.position[2]) * np.inf if self.position[0] == 0 else self.position[2]/self.position[0]
        # slope of drone speed
        tan_phi_point = np.sign(self.speed[2]) * np.inf if self.speed[0] == 0 else self.speed[2]/self.speed[0]
        # slope of forces
        f_ratio = self.drone_model.Fxy / self.drone_model.Fz_minus
        # go up if speed slope is too steep and vertical speed < 0 else take the position angle for forces angle
        psy = 0.5 if tan_phi_point < -tan_phi else -np.arctan(tan_phi * f_ratio) / np.pi + 0.5

        if Settings.perimeter_z / 2 < self.position[2]:
            psy = min(0.2, psy)

        if self.position[0] < 1.5 * Settings.groundzone:
            psy = min(0.2, psy)

        action = np.array([1, theta, psy])

        return action
