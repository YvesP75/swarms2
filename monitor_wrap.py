from dataclasses import make_dataclass

import numpy as np
import gym
import pandas as pd

from settings import Settings
import param_

Path = make_dataclass("Path", [('path', list), ('step', int), ('d_index', int), ('color', list)])


class MonitorWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env, steps):
        # Call the parent constructor, so we can access self.env later
        super(MonitorWrapper, self).__init__(env)
        self.blue_data = []
        self.red_data = []
        self.fire_paths = []
        lat, lon = Settings.latlon
        self.lat_tg = lat
        self.lon_tg = lon
        self.steps = steps
        self.step_ = 0
        self.step_max = 0

    def reset(self):
        """
        Reset the environment
        """
        obs = self.env.reset()
        self.blue_data = []
        self.red_data = []
        self.fire_paths = []
        self.step_ = 0
        self.step_max = 0

        return obs

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """

        obs, reward, done, info = self.env.step(action)

        self.monitor_state()

        self.step_ += 1
        if self.step_ == self.steps:
            done = True

        return obs, reward, done, info

    def monitor_state(self):

        env = self.env
        lat_tg, lon_tg = self.lat_tg, self.lon_tg

        for d_index, drone in enumerate(env.blue_team.drones):
            lat, lon, zed = drone.to_lat_lon_zed(lat_tg, lon_tg)
            self.blue_data.append([self.step_, True, drone.id_, lat, lon, zed, drone.color])

        for d_index, drone in enumerate(env.red_team.drones):
            lat, lon, zed = drone.to_lat_lon_zed(lat_tg, lon_tg)
            self.red_data.append([self.step_, False, drone.id_, lat, lon, zed, drone.color])

        for blue_id, red_id in np.argwhere(0 < env.playground.blues_have_fired_reds):
            b_lat, b_lon, b_zed = env.blue_team.drones[blue_id].to_lat_lon_zed(lat_tg, lon_tg)
            r_lat, r_lon, r_zed = env.red_team.drones[red_id].to_lat_lon_zed(lat_tg, lon_tg)
            self.fire_paths.append(Path(step=self.step_,
                                   path=[[b_lat, b_lon, b_zed], [r_lat, r_lon, r_zed]],
                                   color=param_.GREEN_COLOR,
                                   d_index=blue_id))

        for red_id, blue_id in np.argwhere(0 < env.playground.reds_have_fired_blues):
            b_lat, b_lon, b_zed = env.blue_team.drones[blue_id].to_lat_lon_zed(lat_tg, lon_tg)
            r_lat, r_lon, r_zed = env.red_team.drones[red_id].to_lat_lon_zed(lat_tg, lon_tg)
            self.fire_paths.append(Path(step=self.step_,
                                   path=[[b_lat, b_lon, b_zed], [r_lat, r_lon, r_zed]],
                                   color=param_.BLACK_COLOR,
                                   d_index=red_id))

    def get_df(self):

        fire_df = pd.DataFrame(self.fire_paths)

        df_columns = ['step', 'isBlue', 'd_index', 'lat', 'lon', 'zed', 'color']
        blue_df = pd.DataFrame(self.blue_data, columns=df_columns)
        red_df = pd.DataFrame(self.red_data, columns=df_columns)

        blue_path_df = []
        red_path_df = []

        for d_index in range(self.env.nb_blues):
            blue_path_df.append(self._get_path_df(blue_df, d_index, color=param_.BLUE_COLOR))
        for d_index in range(self.env.nb_reds):
            red_path_df.append(self._get_path_df(red_df, d_index, color=param_.RED_COLOR))

        return blue_df, red_df, fire_df, blue_path_df, red_path_df

    def _get_path_df(self, drone_df: pd.DataFrame, d_index: int, color: int = param_.BLUE_COLOR) -> pd.DataFrame:

        traj_length = param_.TRAJ_LENGTH

        path_total = drone_df[['lon', 'lat', 'zed', 'step']][drone_df.d_index == d_index].values.tolist()
        path = ([Path(path_total[:step+1], step, d_index, color) if step < traj_length
                 else Path(path_total[step - traj_length:step+1], step, d_index, color)
                 for step in range(len(path_total))])
        path_df = pd.DataFrame(path, columns=['path', 'step', 'd_index', 'color'])
        return path_df
