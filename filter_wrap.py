import gym
import numpy as np
from gym import spaces


class FilterWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        # Call the parent constructor, so we can access self.env later

        self.nb_blues, self.nb_reds = env.nb_blues, env.nb_reds

        self.blue_deads = np.full((self.nb_blues,), False)
        self.red_deads = np.full((self.nb_reds,), False)

        env.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=1, shape=(self.nb_blues, 6), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(self.nb_reds, 6), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(self.nb_blues, self.nb_reds), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(self.nb_reds, self.nb_blues), dtype=np.float32),
            spaces.Discrete(1),
            spaces.Discrete(1)))

        super(FilterWrapper, self).__init__(env)

    def reset(self):
        """
        Reset the environment
        """
        obs = self.env.reset()

        return self._sort_obs(obs)

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """

        blue_action, red_action = action

        new_ba = []
        index = 0
        for count, alive in enumerate(~self.blue_deads):
            if alive:
                new_ba.append(blue_action[index])
                index += 1
            else:
                new_ba.append(np.array([0, 0, 0]))
        blue_action = new_ba

        new_ra = []
        index = 0
        for count, alive in enumerate(~self.red_deads):
            if alive:
                new_ra.append(red_action[index])
                index += 1
            else:
                new_ra.append(np.array([0, 0, 0]))
        red_action = new_ra

        action = blue_action, red_action

        obs, reward, done, info = self.env.step(action)

        obs = self._sort_obs(obs)

        return obs, reward, done, info

    def _sort_obs(self, obs):

        blue_obs, red_obs, blues_fire, reds_fire, blue_deads, red_deads = obs

        self.blue_deads = blue_deads
        self.red_deads = red_deads

        blue_obs = np.vstack((blue_obs[~self.blue_deads], blue_obs[self.blue_deads]))
        red_obs = np.vstack((red_obs[~self.red_deads], red_obs[self.red_deads]))

        blues_fire = self.fire_sort(self.blue_deads, self.red_deads, blues_fire)
        reds_fire = self.fire_sort(self.red_deads, self.blue_deads, reds_fire)

        sort_obs = blue_obs, red_obs, blues_fire, reds_fire, sum(blue_deads), sum(red_deads)

        return sort_obs

    def fire_sort(self, dead_friends, dead_foes, friends_fire):

        friends_fire_big = np.zeros_like(friends_fire)
        friends_fire = np.compress(~dead_friends, friends_fire, axis=0)
        friends_fire = np.compress(~dead_foes, friends_fire, axis=1)
        friends_fire_big[:friends_fire.shape[0], :friends_fire.shape[1]] = friends_fire
        return friends_fire_big
