import gym
from gym import spaces
import numpy as np


class ReduxWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env, minus_blue=0, minus_red=0):
        super(ReduxWrapper, self).__init__(env)

        if minus_red == 1:
            alpha = 0

        # action space is reduced
        nb_blues, nb_reds = self.env.nb_blues, self.env.nb_reds

        self.nb_blues = nb_blues - minus_blue
        self.nb_reds = nb_reds - minus_red

        self.blue_deads = minus_blue
        self.red_deads = minus_red

        env.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=1, shape=(self.nb_blues, 6), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(self.nb_reds, 6), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(self.nb_blues, self.nb_reds), dtype=int),
            spaces.Box(low=0, high=1, shape=(self.nb_reds, self.nb_blues), dtype=int)))

        env.action_space = spaces.Tuple((
            spaces.Box(low=0, high=1, shape=(self.nb_blues, 3), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(self.nb_reds, 3), dtype=np.float32)))


    def reset(self):
        """
        Reset the environment
        """
        obs1, obs2, obs3, obs4, obs5, obs6 = self.env.reset()
        return obs1, obs2, obs3, obs4

    def step(self, action):

        # action needs expansion

        blue_action, red_action = action
        if self.blue_deads:
            blue_action = np.vstack((blue_action, np.zeros((self.blue_deads, 3))))
        if self.red_deads:
            red_action = np.vstack((red_action, np.zeros((self.red_deads, 3))))
        action = blue_action, red_action

        obs, reward, done, info = self.env.step(action)

        self.reduce_(obs)

        return obs, reward, done, info

    def reduce_(self, obs):

        # obs need reduction
        blue_obs, red_obs, blues_fire, reds_fire = obs

        if not self.blue_deads:
            pass
        else:
            blue_obs = blue_obs[:-self.blue_deads]
            blues_fire = blues_fire[:-self.blue_deads]
            reds_fire = reds_fire[:, :-self.blue_deads]

        if not self.red_deads:
            pass
        else:
            red_obs = red_obs[:-self.red_deads]
            reds_fire = reds_fire[:-self.red_deads]
            blues_fire = blues_fire[:, :-self.red_deads]

        return blue_obs, red_obs, blues_fire, reds_fire


