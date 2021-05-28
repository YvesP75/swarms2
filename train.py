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
from reward_wrap import RewardWrapper

from settings import Settings
from swarmenv import SwarmEnv
import param_


def bi_train(blue_model, red_model, blues: int = 1, reds: int = 1,
             blue_dispersion: np.float32 = 1, red_dispersion: np.float32 = 1, total_timesteps: int = 1000):
    # If needed create save dir
    save_dir = "policies/" + Settings.policy_folder + f"/b{blues}r{reds}/"
    save_last_dir = "policies/last" + f"/b{blues}r{reds}/"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_last_dir, exist_ok=True)

    # set the dispersion to initial drone positions
    Settings.blue_distance_factor = blue_dispersion * Settings.blue_distance_factor
    Settings.red_distance_factor = red_dispersion * Settings.red_distance_factor
    Settings.red_theta_noise = red_dispersion * Settings.red_theta_noise
    Settings.red_rho_noise = red_dispersion * Settings.red_rho_noise

    # launch learning for red drones and then blue drones
    red_model.learn(total_timesteps=total_timesteps)
    mean_reward, std_reward = evaluate_policy(red_model, red_model.env, n_eval_episodes=10)
    print(f"REDS b{blues}r{reds} disp_b:{blue_dispersion} disp_r{red_dispersion}: mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    red_model.save(save_dir + f"reds_b{10 * blue_dispersion:2.0f}r{10 * red_dispersion:2.0f}")
    red_model.save(save_last_dir + "reds_last")

    blue_model.learn(total_timesteps=total_timesteps)
    mean_reward, std_reward = evaluate_policy(blue_model, blue_model.env, n_eval_episodes=10)
    print(f"BLUES b{blues}r{reds} disp_b:{blue_dispersion} disp_r{red_dispersion}: mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    blue_model.save(save_dir + f"blues_{10 * blue_dispersion:2.0f}r{10 * red_dispersion:2.0f}")
    blue_model.save(save_last_dir + "blues_last")

    return blue_model, red_model


def meta_train(blues: int = 1, reds: int = 1,
               max_dispersion: np.float32 = 3, iteration: int = 10,
               total_timesteps: int = 100):
    Settings.blues, Settings.reds = blues, reds

    # launch the episode to get the data
    steps = int(param_.DURATION / param_.STEP)

    env = SortWrapper(
        SymetryWrapper(
            RotateWrapper(
                ReduxWrapper(
                    DistriWrapper(
                        FilterWrapper(
                            MonitorWrapper(
                                SwarmEnv(blues=blues, reds=reds), steps, verbose=False)))))))

    blue_env = RewardWrapper(TeamWrapper(env, is_blue=True), is_blue=True)
    red_env = RewardWrapper(TeamWrapper(env, is_blue=False), is_blue=False)

    blue_model = SAC(MlpPolicy, blue_env, verbose=0)
    red_model = SAC(MlpPolicy, red_env, verbose=0)

    for red_dispersion in np.linspace(0.3, max_dispersion, num=iteration):
        for blue_dispersion in np.linspace(max_dispersion, 0.3, num=iteration):
            blue_model, red_model = bi_train(
                blue_model, red_model, blues=blues, reds=reds,
                blue_dispersion=blue_dispersion, red_dispersion=red_dispersion,
                total_timesteps=total_timesteps)


def super_meta_train(max_blues: int = 3, max_reds: int = 3, max_dispersion: np.float32 = 3,
                     iteration: int = 10, total_timesteps: int = 100, policy_folder: str = "default"):
    Settings.policy_folder = policy_folder
    for drones_nb in range(2, max_blues + max_reds + 1):
        for blues in range(1, max_blues + 1):
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


super_meta_train(max_blues=1, max_reds=1, iteration=6, max_dispersion=2, total_timesteps=5000, policy_folder="0528_10")
# super_meta_train(max_blues=2, max_reds=2, iteration=4, max_dispersion=3, total_timesteps=10, policy_folder="0528_test")
