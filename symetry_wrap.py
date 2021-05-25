import numpy as np
import gym


class SymetryWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        # Call the parent constructor, so we can access self.env later

        self.symetry = False  # no need to perform a symetry
        super(SymetryWrapper, self).__init__(env)

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
        if self.symetry:
            action = symetrise_action(action)

        obs, reward, done, info = self.env.step(action)

        obs = self.post_obs(obs)

        return obs, reward, done, info

    def post_obs(self, obs):
        self.symetry = get_symetry(obs)
        if self.symetry:
            obs = symetrise_obs(obs)
        return obs


def get_symetry(obs):
    blue_obs, red_obs, blue_fire, red_fire = obs

    # count the drones who are positioned above the 0 x-axis
    count = 0
    for this_obs in (blue_obs, red_obs):
        for d in this_obs:
            add = 1 if (d[1] < 0.5) else 0
            count += add

    # compare with the total
    symetry = bool(2*count < (len(blue_obs) + len(red_obs)))

    return symetry


def symetrise_obs(obs):

    blue_obs, red_obs, blue_fire, red_fire = obs

    for this_obs in (blue_obs, red_obs):
        # symetrise positions and speeds
        this_obs[:, 1] = 1 - this_obs[:, 1]
        this_obs[:, 4] = 1 - this_obs[:, 4]

    return blue_obs, red_obs, blue_fire, red_fire


def symetrise_action(action):

    blue_action, red_action = action

    for this_action in (blue_action, red_action):
        for act in this_action:

            # symetrise action
            act[1] = - act[1]

    action = blue_action, red_action

    return action


def test_symetrise_obs():

    obs = np.arange(12).reshape(2, 6), np.arange(12).reshape(2, 6), np.random.random((1, 1)), np.random.random((1, 1))
    print(obs)
    symetrise_obs(obs)
    print(obs)
