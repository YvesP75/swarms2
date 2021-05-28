import gym
from gym import spaces
import numpy as np

import param_
from settings import Settings
from playground import Playground
from team import BlueTeam, RedTeam


class SwarmEnv(gym.Env):
    """
    Custom 3D-Environment that follows gym interface.
    This is a 3D-env where the blue drones defend a circular GROUNDZONE from a red drones attack
    """

    def __init__(self, blues=Settings.blues, reds=Settings.reds):
        """
        :param distance: the distance to the other rim
        """
        super(SwarmEnv, self).__init__()

        self.nb_blues = blues
        self.nb_reds = reds

        self.blue_team = BlueTeam(number_of_drones=self.nb_blues)
        self.red_team = RedTeam(number_of_drones=self.nb_reds)

        self.playground = Playground(env=self, blue_drones=self.blue_team.drones, red_drones=self.red_team.drones)

        self.steps = 0

        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=1, shape=(self.nb_blues, 6), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(self.nb_reds, 6), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(self.nb_blues, self.nb_reds), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(self.nb_reds, self.nb_blues), dtype=np.float32),
            spaces.MultiBinary(self.nb_blues),
            spaces.MultiBinary(self.nb_reds),
        ))

        self.action_space = spaces.Tuple((
            spaces.Box(low=0, high=1, shape=(self.nb_blues, 3), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(self.nb_reds, 3), dtype=np.float32)))

    def reset(self, obs=None):
        """
        resets the environment as part of Gym interface
        """
        if obs:
            blue_obs, red_obs, blues_fire, reds_fire, blue_deads, red_deads = obs
        else:
            blue_obs, red_obs, blues_fire, reds_fire, blue_deads, red_deads = None, None, None, None, None, None

        self.blue_team.reset(obs=blue_obs)
        self.red_team.reset(obs=red_obs)
        self.playground.reset()
        self.steps = 0

        # get observations from blue and red teams
        blue_obs, blue_deads = self.blue_team.get_observation()
        red_obs, red_deads = self.red_team.get_observation()
        blues_fire, reds_fire = self.playground.get_observation()

        return blue_obs, red_obs, blues_fire, reds_fire, blue_deads, red_deads

    def render(self, mode='human'):
        pass

    def step(self, action):
        blue_action, red_action = action
        blue_obs, blue_reward, blue_done, blue_info = self.blue_team.step(blue_action)
        red_obs, red_reward, red_done, red_info = self.red_team.step(red_action)
        bf_obs, bf_reward, bf_done, bf_info, rf_obs, rf_reward, rf_done, rf_info = self.playground.step()
        _, blue_deads = self.blue_team.get_observation()
        _, red_deads = self.red_team.get_observation()
        obs = blue_obs, red_obs, bf_obs, rf_obs, blue_deads, red_deads
        reward = blue_reward + red_reward + bf_reward + rf_reward
        done = False

        info = {}
        info['red_oob'] = red_info['oob']
        info['blue_oob'] = blue_info['oob']
        info['hits_target'] = red_info['hits_target']
        info['blue_shots'] = rf_info
        info['red_shots'] = bf_info
        info['weighted_red_distance'] = red_info['delta_distance']
        info['remaining reds'] = bf_done
        info['remaining blues'] = rf_done
        info['ttl'] = red_info['ttl']
        info['distance_to_straight_action'] = red_info['distance_to_straight_action']


        return obs, reward, done, info
