import numpy as np
import gym
from gym import spaces

from swarm_policy import SwarmPolicy
from settings import Settings


class TeamWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env, is_blue: bool = True, is_double: bool = False, is_unkillable: bool = Settings.is_unkillable):

        self.is_blue = is_blue
        self.is_double = is_double
        self.is_unkillabe = is_unkillable

        nb_blues, nb_reds = env.nb_blues, env.nb_reds

        self.foe_action = None
        self.foe_policy = SwarmPolicy(is_blue=not is_blue, blues=nb_blues, reds=nb_reds)

        if is_double:
            env.action_space = spaces.Tuple((
                spaces.Box(low=0, high=1, shape=(nb_blues*3,), dtype=np.float32),
                spaces.Box(low=0, high=1, shape=(nb_reds*3,), dtype=np.float32)))
        else:
            nb_friends = nb_blues if is_blue else nb_reds
            env.action_space = spaces.Box(low=0, high=1, shape=(nb_friends*3,), dtype=np.float32)

        flatten_dimension = 6 * nb_blues + 6 * nb_reds
        flatten_dimension += (nb_blues * nb_reds) * (1 if is_unkillable else 2)

        env.observation_space = spaces.Box(low=-1, high=1, shape=(flatten_dimension,), dtype=np.float32)

        super(TeamWrapper, self).__init__(env)

    def reset(self):
        """
        Reset the environment
        """
        obs = self.env.reset()
        obs = self.post_obs(obs)

        return obs

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        if self.is_double:
            blue_action, red_action = action
            action = _unflatten(blue_action), _unflatten(red_action)
        else:
            if self.is_blue:
                action = _unflatten(action), _unflatten(self.foe_action)
            else:
                action = _unflatten(self.foe_action), _unflatten(action)

        obs, reward, done, info = self.env.step(action)

        obs = self.post_obs(obs)

        if not self.is_blue:
            reward = - reward

        return obs, reward, done, info

    def post_obs(self, obs):

        if self.is_unkillabe:
           o1, o2, o3, _ = obs
           obs = o1, o2, o3
        flatten_obs = _flatten(obs)
        centralised_obs = _centralise(flatten_obs)

        if not self.is_double:
            self.foe_action = self.foe_policy.predict(centralised_obs)

        return centralised_obs


def _unflatten(action):
    return np.split(action, len(action)/3)


def _flatten(obs):  # need normalisation too
    fl_obs = [this_obs.flatten().astype('float32') for this_obs in obs]
    fl_obs = np.hstack(fl_obs)
    return fl_obs


def _centralise(obs):  # [0,1] to [-1,1]
    obs = 2 * obs - 1
    return obs
