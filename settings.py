from dataclasses import dataclass

import param_
import streamlit as st
import numpy as np


@dataclass
class Settings:

    perimeter: int = param_.PERIMETER
    perimeter_z: int = param_.PERIMETER_Z
    groundzone: int = param_.GROUNDZONE

    latlon = param_.LATLON['Paris']['lat'], param_.LATLON['Paris']['lon']

    blues: int = param_.BLUES

    blues_per_circle = np.array(param_.BLUES_PER_CIRCLE)
    blue_circles_rho = param_.BLUE_CIRCLES_RHO
    blue_circles_theta = param_.BLUE_CIRCLES_THETA
    blue_circles_zed = param_.BLUE_CIRCLES_ZED
    blue_distance_factor: float = param_.BLUE_DISTANCE_FACTOR

    is_unkillable: bool = param_.BLUE_IS_UNKILLABLE

    blue_speed_init: int = param_.BLUE_SPEED_INIT

    reds: int = param_.REDS

    red_squads = param_.RED_SQUADS
    red_squads_rho = np.array(param_.RED_SQUADS_RHO)
    red_squads_theta = param_.RED_SQUADS_THETA
    red_squads_zed = param_.RED_SQUADS_ZED
    red_distance_factor: float = param_.RED_DISTANCE_FACTOR

    red_rho_noise = np.array(param_.RED_RHO_NOISE)
    red_theta_noise = np.array(param_.RED_THETA_NOISE)
    red_zed_noise = np.array(param_.RED_ZED_NOISE)

    red_speed_init: int = param_.RED_SPEED_INIT

    policy_folder: str = param_.POLICY_FOLDER


def define_(with_streamlit: bool = True, blues: int = Settings.blues, reds: int = Settings.reds):
    """"
        shows the blue and red swarms in Streamlit
        :return:
        """
    blues = blues
    reds = reds

    if with_streamlit:
        st.title('Blues against Reds by hexamind.ai')
        st.write('controlled by Reinforcement Learning.')
        st.text('<- Set parameters')

        st.sidebar.subheader("Define the battlefield")
        blues = st.sidebar.slider("how many blues on defense?", 1, 12, 6)
        Settings.blues = blues
        blue_dispersion = st.sidebar.slider("set the average blue dispersion", 0.3, 1.0, 0.8)
        Settings.reds = reds
        reds = st.sidebar.slider("how many reds are on the attack?", 1, 12, 4)
        Settings.reds = reds
        red_dispersion = st.sidebar.slider("set the average red dispersion", 0.3, 1.0, 0.7)

        Settings.blue_distance_factor = 3 * blue_dispersion
        Settings.red_distance_factor = 3 * red_dispersion

        location = st.sidebar.radio("Location", ['Paris', 'Puilaurens', 'San Francisco'])

        lat_tg = param_.LATLON[location]['lat']
        lon_tg = param_.LATLON[location]['lon']

        Settings.latlon = lat_tg, lon_tg

        st.sidebar.write(
            'you probably need more drones '
            'No worries, we have plenty at www.hexamind.ai ')

    return blues, reds
