import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
import os

from monitor_wrap import MonitorWrapper
from filter_wrap import FilterWrapper
from distribution_wrap import DistriWrapper
from redux_wrap import ReduxWrapper
from symetry_wrap import SymetryWrapper
from rotate_wrap import RotateWrapper
from sort_wrap import SortWrapper
from team_wrap import TeamWrapper

from settings import Settings
from swarmenv import SwarmEnv


def bi_train(blue_model, red_model, blues: int = 1, reds: int = 1,
             dispersion: np.float32 = 1, total_timesteps: int = 1000):
    # Create save dir
    save_dir = "policies/" + Settings.policy_folder + f"/b{blues}r{reds}/"
    os.makedirs(save_dir, exist_ok=True)

    # set the dispersion to initial drone positions
    Settings.blue_distance_factor = dispersion * Settings.blue_distance_factor
    Settings.red_distance_factor = dispersion * Settings.red_distance_factor
    Settings.red_theta_noise = dispersion * Settings.red_theta_noise
    Settings.red_rho_noise = dispersion * Settings.red_rho_noise

    # launch learning for blue drones and then red drones
    blue_model.learn(total_timesteps=total_timesteps)
    mean_reward, std_reward = evaluate_policy(blue_model, blue_model.env, n_eval_episodes=10)
    print(f"BLUES : mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    blue_model.save(save_dir + f"blues_{dispersion:.0f}")
    blue_model.save(save_dir + f"blues_last")

    red_model.learn(total_timesteps=total_timesteps)
    mean_reward, std_reward = evaluate_policy(red_model, red_model.env, n_eval_episodes=10)
    print(f"REDS : mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    red_model.save(save_dir + f"reds_{dispersion:.0f}")
    red_model.save(save_dir + f"reds_last")

    return blue_model, red_model


def meta_train(blues: int = 1, reds: int = 1,
               max_dispersion: np.float32 = 3, iteration: int = 10,
               total_timesteps: int = 100):

    Settings.blues, Settings.reds = blues, reds

    env = SortWrapper(
        SymetryWrapper(
            RotateWrapper(
                ReduxWrapper(
                    DistriWrapper(
                        FilterWrapper(
                            MonitorWrapper(
                                SwarmEnv(blues=blues, reds=reds), -1, verbose=False)))))))

    blue_env = TeamWrapper(env, is_blue=True)
    red_env = TeamWrapper(env, is_blue=False)

    blue_model = SAC(MlpPolicy, blue_env, verbose=0)
    red_model = SAC(MlpPolicy, red_env, verbose=0)
    for dispersion in np.linspace(1, max_dispersion, num=iteration):
        blue_model, red_model = bi_train(
            blue_model, red_model, blues=blues, reds=reds, dispersion=dispersion, total_timesteps=total_timesteps)


def super_meta_train(max_blues: int = 3, max_reds: int = 3, max_dispersion: np.float32 = 3,
                     iteration: int = 10, total_timesteps: int = 100):
    for drones_nb in range(2, max_blues+max_reds+1):
        for blues in range(1, max_blues+1):
            reds = drones_nb - blues
            if 1 <= reds <= max_reds:
                print(f"reds :{reds}, blues: {blues}")
                meta_train(blues=blues, reds=reds,
                           max_dispersion=max_dispersion, iteration=iteration, total_timesteps=total_timesteps)


def print_spaces(env, name: str):
    print("++++++++++++")
    print(name)
    print(env.action_space)
    print(env.observation_space)
    print("============")
    check_env(env, warn=True)


# meta_train(iteration=3, total_timesteps=100)

#  super_meta_train(max_blues=2, max_reds=2, iteration=2, max_dispersion=2, total_timesteps=10)
super_meta_train(max_blues=6, max_reds=6, iteration=12, max_dispersion=3, total_timesteps=10000)
