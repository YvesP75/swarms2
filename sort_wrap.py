import numpy as np
import gym


class SortWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        # Call the parent constructor, so we can access self.env later
        super(SortWrapper, self).__init__(env)
        self.blue_signature = None
        self.red_signature = None

    def reset(self):
        """
        Reset the environment
        """
        obs = self.env.reset()
        obs = self.sort_obs(obs)

        return obs

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """

        action = self.unsort_action(action)

        obs, reward, done, info = self.env.step(action)

        obs = self.post_obs(obs)

        return obs, reward, done, info

    def post_obs(self, obs):
        return self.sort_obs(obs)

    def sort_obs(self, obs):

        blue_obs, red_obs, blue_fire, red_fire = obs

        blue_obs, self.blue_signature = self.sort_and_sign(blue_obs)
        red_obs, self.red_signature = self.sort_and_sign(red_obs)

        blue_fire = self.unsort_matrix_with_signatures(blue_fire, self.blue_signature, self.red_signature)
        red_fire = self.unsort_matrix_with_signatures(red_fire, self.red_signature, self.blue_signature)

        obs = blue_obs, red_obs, blue_fire, red_fire

        return obs

    def unsort_action(self, action):

        blue_action, red_action = action

        unsorted_blue_action = self.unsort_with_signature(blue_action, self.blue_signature)
        unsorted_red_action = self.unsort_with_signature(red_action, self.red_signature)

        action = unsorted_blue_action, unsorted_red_action

        return action

    def sort_and_sign(self, an_array: np.ndarray) -> [np.ndarray, []]:
        """
        allows to sort an ndarray of 6 columns of floats and to keep the "signature": the positions of the items
        before being sorted in order to retrieve the initial order after modifications of the arrays.
        the order is retrieved with the unsort_with_signature function
        :param an_array:
        :return:
        """
        zip_list = zip(an_array, range(len(an_array)))
        zip_sorted = sorted(zip_list, key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3], x[0][4], x[0][5]))
        sorted_array, signature = zip(*zip_sorted)
        return np.array(sorted_array), signature

    def unsort_with_signature(self, an_array: np.ndarray, signature: []) -> np.ndarray:
        """
        see above
        :param an_array:
        :param signature:
        :return:
        """
        zip_list = zip(signature, an_array)
        zip_unsorted = sorted(zip_list)
        _, unsorted = zip(*zip_unsorted)
        return np.array(unsorted)

    def unsort_matrix_with_signatures(self, matrix: np.ndarray, sign_line: np.ndarray, sign_col: np.ndarray) \
            -> np.ndarray:

        matrix = self.unsort_with_signature(matrix, sign_line)
        matrix = self.unsort_with_signature(matrix.T, sign_col).T

        return matrix
