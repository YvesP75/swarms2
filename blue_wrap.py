import numpy as np
import gym
from gym import spaces

from swarm_policy import SwarmPolicy


class BlueWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        # Call the parent constructor, so we can access self.env later
        super(BlueWrapper, self).__init__(env)
        self.red_action = None
        self.nb_blues, self.nb_reds = env.nb_blues, env.nb_reds
        self.red_policy = SwarmPolicy(is_blue=False, blues=self.nb_blues, reds=self.nb_reds)

        env.action_space = spaces.Box(low=0, high=1, shape=(self.nb_blues, 3), dtype=np.float32)

    def reset(self):
        """
        Reset the environment
        """
        obs = self.env.reset()
        self.red_action = self.red_policy.predict(obs)

        return obs

    def step(self, blue_action):
        """
        :param blue_action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """

        action = blue_action, self.red_action

        obs, reward, done, info = self.env.step(action)

        self.red_action = self.red_policy.predict(obs)

        return obs, reward, done, info
