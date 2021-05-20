
from swarm_policy import SwarmPolicy


def run_episode(env, obs):
    blue_obs, red_obs, blues_fire, reds_fire = obs
    blues = len(blue_obs)
    reds = len(red_obs)
    blue_policy = SwarmPolicy(blues=blues, reds=reds, is_blue=True)
    red_policy = SwarmPolicy(blues=blues, reds=reds, is_blue=False)
    sum_reward = 0
    done = False
    while not done:
        action = blue_policy.predict(obs), red_policy.predict(obs)
        obs, reward, done, info = env.step(action)
        sum_reward += reward
    return obs, sum_reward, done, info


