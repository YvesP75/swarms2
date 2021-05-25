import gym

from gym import spaces
import numpy as np

from runner import run_episode
from redux_wrap import ReduxWrapper
from rotate_wrap import RotateWrapper
from symetry_wrap import SymetryWrapper
from sort_wrap import SortWrapper
from team_wrap import TeamWrapper


class DistriWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):

        self.blue_deads = self.red_deads = 0
        self.nb_blues, self.nb_reds = env.nb_blues, env.nb_reds

        env.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=1, shape=(self.nb_blues, 6), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(self.nb_reds, 6), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(self.nb_blues, self.nb_reds), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(self.nb_reds, self.nb_blues), dtype=np.float32)))

        env.action_space = spaces.Tuple((
            spaces.Box(low=0, high=1, shape=(self.nb_blues, 3), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(self.nb_reds, 3), dtype=np.float32)))

        # Call the parent constructor, so we can access self.env later
        super(DistriWrapper, self).__init__(env)

    def reset(self):
        """
        Reset the environment
        """
        obs = self.env.reset()
        blue_obs, red_obs, blues_fire, reds_fire, blue_deads, red_deads = obs
        self.blue_deads, self.blue_deads = blue_deads, red_deads
        return blue_obs, red_obs, blues_fire, reds_fire

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
            return obs, reward, True, info

        # do we have new deaths?
        new_blue_deads = blue_deads - self.blue_deads
        new_red_deads = red_deads - self.red_deads
        self.blue_deads, self.red_deads = blue_deads, red_deads

        if 0 < new_red_deads + new_blue_deads:  # we have someone killed but we still have some fight

            blues, reds = self.nb_blues - blue_deads, self.nb_reds - red_deads

            env = ReduxWrapper(self,  minus_blue=blue_deads, minus_red=red_deads)
            obs_ = env.post_obs(obs)

            env = RotateWrapper(env)
            obs_ = env.post_obs(obs_)

            env = SymetryWrapper(env)
            obs_ = env.post_obs(obs_)

            env = SortWrapper(env)
            obs_ = env.post_obs(obs_)

            env = TeamWrapper(env, is_double=True)
            obs_ = env.post_obs(obs_)

            _, reward, done, info = run_episode(env, obs_, blues=blues, reds=reds)

        return obs, reward, done, info
