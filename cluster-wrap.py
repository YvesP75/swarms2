import gym
import numpy as np

class ClusterWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        # Call the parent constructor, so we can access self.env later
        super(ClusterWrapper, self).__init__(env)
        self.red_clusters = [[0, 1, 2, 3, 4]]
        self.blue_clusters = [[0, 1, 2]]
        self.learner_is_blue = True
        self.meta_learner = False # it is a drone team which learns

    def reset(self):
        """
        Reset the environment
        """
        obs = self.env.reset()
        return obs

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        obs, reward, done, info = self.env.step(action)

        # create clusters based on the obs
        self.clusters = self.get_clusters(obs)

        # define the obs per cluster
        clustered_obs = self.get_clustered_obs(obs)

        # get actions per color from clusters except cluster 0 where we learn
        other_clusters = self.clusters[1:]
        other_obs = clustered_obs[1:]
        other_actions = [self.get_actions(cluster, cluster_obs)
                         for cluster, cluster_obs in zip(other_clusters, other_obs)]
        if self.learner_is_blue:
            self.get_red_actions(clustered_obs[0])
        else:
            pass

        return obs, reward, done, info
