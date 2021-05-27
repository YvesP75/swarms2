import numpy as np
import gym
from gym import spaces

import param_
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
        self.info = {}

        nb_blues, nb_reds = env.nb_blues, env.nb_reds

        self.foe_action = None
        self.foe_policy = SwarmPolicy(is_blue=not is_blue, blues=nb_blues, reds=nb_reds)

        if is_double:
            env.action_space = spaces.Tuple((
                spaces.Box(low=-1, high=1, shape=(nb_blues*3,), dtype=np.float32),
                spaces.Box(low=-1, high=1, shape=(nb_reds*3,), dtype=np.float32)
            ))
        else:
            nb_friends = nb_blues if is_blue else nb_reds
            env.action_space = spaces.Box(low=-1, high=1, shape=(nb_friends*3,), dtype=np.float32)

        flatten_dimension = 6 * nb_blues + 6 * nb_reds  # the position and speeds for blue and red drones
        flatten_dimension += (nb_blues * nb_reds) * (1 if is_unkillable else 2)  # the fire matrices

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
            blue_action = _decentralise(blue_action)
            red_action = _decentralise(red_action)
            action = _unflatten(blue_action), _unflatten(red_action)
        else:
            friend_action = _decentralise(action)
            foe_action = _decentralise(self.foe_action)
            if self.is_blue:
                action = _unflatten(friend_action), _unflatten(foe_action)
            else:
                action = _unflatten(foe_action), _unflatten(friend_action)

        obs, reward, done, info = self.env.step(action)

        obs = self.post_obs(obs)

        reward, done, info = self.situation_evaluation(reward, info)

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

    def situation_evaluation(self, reward, info):

        if self.is_double:
            if info['red_loses'] or info['blue_loses']:
                return 0, True, info
            else:
                return 0, False, info

        else:
            if self.is_blue:
                if info['red_loses']:
                    return param_.WIN_REWARD, True, info
                if info['blue_loses']:
                    return -param_.WIN_REWARD, True, info
                if 0 < info['blue_oob']:
                    return -param_.OOB_COST, True, info
                # else continues
                reward = -param_.STEP_COST
                reward -= info['weighted_red_distance'] * param_.THREAT_WEIGHT
                reward -= info['hits_target'] * param_.TARGET_HIT_COST
                reward += info['red_shots'] * param_.RED_SHOT_REWARD
                return reward, False, info
            else:  # red is learning
                if info['red_loses']:
                    return -param_.WIN_REWARD, True, info
                if info['blue_loses']:
                    return param_.WIN_REWARD, True, info
                if 0 < info['red_oob']:
                    return -param_.OOB_COST, True, info
                # else continues
                reward = -param_.STEP_COST
                reward += info['weighted_red_distance'] * param_.THREAT_WEIGHT
                reward += info['hits_target'] * param_.TARGET_HIT_COST
                reward -= info['red_shots'] * param_.RED_SHOT_REWARD
                return reward, False, info


def _unflatten(action):
    return np.split(action, len(action)/3)


def _flatten(obs):  # need normalisation too
    fl_obs = [this_obs.flatten().astype('float32') for this_obs in obs]
    fl_obs = np.hstack(fl_obs)
    return fl_obs


def _centralise(obs):  # [0,1] to [-1,1]
    obs = 2 * obs - 1
    return obs


def _decentralise(act):  # [-1,1] to [0,1]
    act = 0.5 * (act + 1)
    return act
