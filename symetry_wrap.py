import numpy as np
import gym

from drone import Drone


class SymetryWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        # Call the parent constructor, so we can access self.env later
        super(SymetryWrapper, self).__init__(env)
        self.symetry = False  # no need to perform a symetry

    def reset(self):
        """
        Reset the environment
        """
        obs = self.env.reset()

        self.symetry = self.get_symetry(obs)
        if self.symetry:
            obs = self.symetrise_obs(obs)

        return obs

    def step(self, action):
        """
        :param blue_action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        if self.symetry:
            action = self.symetrise_action(action)

        obs, reward, done, info = self.env.step(action)

        self.symetry = self.get_symetry(obs)
        if self.symetry:
            obs = self.symetrise_obs(obs)

        return obs, reward, done, info

    def get_symetry(self, obs):
        blue_obs, red_obs, blue_fire, red_fire = obs

        # count the drones who are positioned above the 0 x-axis
        count = 0
        for this_obs in (blue_obs, red_obs):
            for d in this_obs:
                add = 1 if (d[1] < 0.5) else 0
                count += add

        # compare with the total
        symetry = 2*count < (len(blue_obs) + len(red_obs))

        return symetry


    def symetrise_obs(self, obs):

        blue_obs, red_obs, blue_fire, red_fire = obs

        for this_obs in (blue_obs, red_obs):
            for d in this_obs:

                # symetrise positions and speeds
                d[1] = - d[1]
                d[4] = - d[4]

        return blue_obs, red_obs, blue_fire, red_fire


    def symetrise_action(self, action):

        blue_action, red_action = action

        for this_action in (blue_action, red_action):
            for act in this_action:

                # symetrise action
                act[1] = - act[1]

        action = blue_action, red_action

        return action

