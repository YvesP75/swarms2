import gym

from gym import spaces
import numpy as np

from runner import run_episode
from redux_wrap import ReduxWrapper
from rotate_wrap import RotateWrapper
from symetry_wrap import SymetryWrapper


class DistriWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        # Call the parent constructor, so we can access self.env later
        super(DistriWrapper, self).__init__(env)
        self.blue_deads = self.red_deads = 0
        self.nb_blues, self.nb_reds = env.nb_blues, env.nb_reds

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
        obs = self.env.reset()
        blue_obs, red_obs, blues_fire, reds_fire, blue_deads, red_deads = obs
        self.blue_deads, self.blue_deads = blue_deads, red_deads
        return obs

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        obs, reward, done, info = self.env.step(action)

        blue_obs, red_obs, blues_fire, reds_fire, blue_deads, red_deads = obs
        obs = blue_obs, red_obs, blues_fire, reds_fire

        if done:  # environment decision (eg drones oob)
            return obs, reward, True, info

        if red_deads == len(red_obs):  # no more reds to fight (it could mean that they have all reached the target)
            return obs, reward, True, info

        if blue_deads == len(blue_obs):  # reds have won
            remaining_reds = len(red_obs) - red_deads
            return obs, reward, True, info

        new_blue_deads = blue_deads - self.blue_deads
        new_red_deads = red_deads - self.red_deads
        self.blue_deads, self.red_deads = blue_deads, red_deads

        if 0 < new_red_deads + new_blue_deads:  # we have someone killed but we still have some fight

            reduced_env = SymetryWrapper(RotateWrapper(ReduxWrapper(self,  minus_blue=blue_deads, minus_red=red_deads)))
            reduced_obs = reduced_env.reduce_(obs)
            obs, reward_episode, _, info = run_episode(reduced_env, reduced_obs)
            return obs, reward_episode, True, info

        return obs, reward, False, info
