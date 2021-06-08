
import gym

import param_
from settings import Settings


class RewardWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env, is_blue: bool = True, is_double: bool = False):

        self.is_blue = is_blue
        self.is_double = is_double

        super(RewardWrapper, self).__init__(env)

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

        reward, done, info = self.situation_evaluation(info)

        return obs, reward, done, info

    def situation_evaluation(self, info):

        if self.is_double:
            if info['remaining blues'] * info['remaining reds'] == 0:
                return 0, True, info
            else:
                return 0, False, info

        else:
            if self.is_blue:
                if info['remaining reds'] == 0:
                    return param_.WIN_REWARD, True, info
                if info['remaining blues'] == 0:
                    return -param_.WIN_REWARD, True, info
                if 0 < info['blue_oob']:
                    return -param_.OOB_COST, True, info
                if info['ttl'] < 0:
                    return -param_.TTL_COST, True, info  # blues have been too long to shoot the red drone
                # else continues
                reward = -param_.STEP_COST
                reward -= info['weighted_red_distance'] * param_.THREAT_WEIGHT
                reward -= info['hits_target'] * param_.TARGET_HIT_COST
                reward += info['red_shots'] * param_.RED_SHOT_REWARD
                reward += info['distance_to_straight_action'] * param_.STRAIGHT_ACTION_COST
                return reward, False, info
            else:  # red is learning
                done = False
                reward = -param_.STEP_COST
                reward += info['weighted_red_distance'] * param_.THREAT_WEIGHT
                reward += info['hits_target'] * param_.TARGET_HIT_COST
                reward -= info['red_shots'] * param_.RED_SHOT_REWARD
                reward -= info['distance_to_straight_action'] * param_.STRAIGHT_ACTION_COST
                if info['remaining reds'] == 0:
                    done = True
                    return reward, done, info
                if info['remaining blues'] == 0:
                    done = True
                    reward += info['remaining reds'] * param_.TARGET_HIT_COST
                    return reward, done, info
                if 0 < info['red_oob']:
                    done = True
                    reward -= param_.OOB_COST
                if info['ttl'] < 0:
                    done = True
                    reward -= param_.TTL_COST * info['remaining reds']  # reds have been too long to hit the target
                # else continues

                return reward, done, info
