import numpy as np
import gym

from drone import Drone


class RotateWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        # Call the parent constructor, so we can access self.env later
        super(RotateWrapper, self).__init__(env)
        self.angle = 0

    def reset(self):
        """
        Reset the environment
        """
        obs = self.env.reset()

        self.angle = self.get_angle(obs)
        obs = self.rotate_obs(obs)

        return obs

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """

        action = self.rotate_action(action)

        obs, reward, done, info = self.env.step(action)


        self.angle = self.get_angle(obs)
        obs = self.rotate_obs(obs)

        return obs, reward, done, info

    def get_angle(self, obs: np.ndarray) -> float:
        blue_obs, red_obs, blue_fire, red_fire = obs
        sigma = 0

        for this_obs in (blue_obs, red_obs):
            for d in this_obs:
                sigma += d[0] * np.exp(1j * d[1])
        angle = np.angle(sigma)
        return angle

    def rotate_obs(self, obs):
        blue_obs, red_obs, blue_fire, red_fire = obs

        rotated_blue_obs = []
        rotated_red_obs = []

        for this_obs, is_blue, rotated_obs in zip((blue_obs, red_obs),
                                                  (True, False),
                                                  (rotated_blue_obs, rotated_red_obs)):
            drone = Drone(is_blue=is_blue)
            for d in this_obs:

                d_meter = np.zeros(6,)
                # get the pos and speed in cylindrical coordinated in meters
                d_meter[:3] = drone.from_norm(d[:3], drone.max_positions, drone.min_positions)
                d_meter[3:6] = drone.from_norm(d[3:6], drone.max_speeds, drone.min_speeds)

                # rotate
                d_meter[1] -= self.angle
                d_meter[4] -= self.angle

                # back to norm
                d[:3] = drone.to_norm(d_meter[:3], drone.max_positions, drone.min_positions)
                d[3:6] = drone.to_norm(d_meter[3:6], drone.max_speeds, drone.min_speeds)

                rotated_obs.append(d)

            del drone

        return np.array(rotated_blue_obs), np.array(rotated_red_obs), blue_fire, red_fire

    def rotate_action(self, action):

        blue_action, red_action = action
        blue_action = np.array(list(map(lambda x: [x[0], (x[1]+self.angle/2/np.pi) % 1, x[2]], blue_action)))
        red_action = np.array(list(map(lambda x: [x[0], (x[1]+self.angle/2/np.pi) % 1, x[2]], red_action)))
        action = blue_action, red_action

        return action
