import time
import pandas as pd
import pydeck as pdk
import streamlit as st

from monitor_wrap import MonitorWrapper
from filter_wrap import FilterWrapper
from distribution_wrap import DistriWrapper
from redux_wrap import ReduxWrapper
from symetry_wrap import SymetryWrapper
from rotate_wrap import RotateWrapper
from sort_wrap import SortWrapper
from team_wrap import TeamWrapper

from runner import run_episode
from settings import Settings, define_
import param_
from swarmenv import SwarmEnv


def run(with_streamlit=True, blues: int = 1, reds: int = 1):

    # define settings with Streamlit (or use default parameters)
    blues, reds = define_(with_streamlit=with_streamlit, blues=blues, reds=reds)

    # put in place the map
    deck_map, initial_view_state = pre_show(with_streamlit=with_streamlit)

    # launch the episode to get the data
    steps = int(param_.DURATION / param_.STEP)
    monitor_env = MonitorWrapper(SwarmEnv(blues=blues, reds=reds), steps)
    env = FilterWrapper(monitor_env)
    env = DistriWrapper(env)
    env = ReduxWrapper(env)
    env = SortWrapper(
            SymetryWrapper(
                RotateWrapper(env)))

    env = TeamWrapper(env, is_double=True)

    obs = env.reset()
    run_episode(env, obs, blues=blues, reds=reds)

    # display the data with Streamlit
    if with_streamlit:
        show(monitor_env, deck_map, initial_view_state)


def pre_show(with_streamlit=True):
    if with_streamlit:
        deck_map = st.empty()
        pitch = st.slider('pitch', 0, 100, 50)
        lat, lon = Settings.latlon
        initial_view_state = pdk.ViewState(
            latitude=lat,
            longitude=lon,
            zoom=13,
            pitch=pitch
        )
        return deck_map, initial_view_state
    else:
        return 0, 0


def show(monitor_env, deck_map, initial_view_state):

    blue_df, red_df, fire_df, blue_path_df, red_path_df = monitor_env.get_df()
    step_max = monitor_env.step_

    for step in range(step_max):
        deck_map.pydeck_chart(pdk.Deck(
            map_provider="mapbox",
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=initial_view_state,
            layers=get_layers(blue_df,
                              red_df,
                              blue_path_df,
                              red_path_df,
                              step)
        ))

        time.sleep(param_.STEP*param_.SIMU_SPEED)


def get_layers(df_blue: pd.DataFrame, df_red: pd.DataFrame,
               df_blue_path: [pd.DataFrame], df_red_path: [pd.DataFrame],
               step) -> [pdk.Layer]:
    lat, lon = Settings.latlon
    df_target = pd.DataFrame({'lat': [lat], 'lon': [lon]})
    layers_ = get_target_layers(df_target)

    for (df, dfp, b) in [(df_blue, df_blue_path, True), (df_red, df_red_path, False)]:
        layers_.append(get_current_drone_layers(df, step))
        nb_drones = df['d_index'].max() + 1
        for drone_index in range(nb_drones):
            layers_.append(get_path_layers(dfp[drone_index], step))

    return layers_


def get_target_layers(df_target) -> [pdk.Layer]:
    return [
        # this is the GROUNDZONE
        pdk.Layer(
            'ScatterplotLayer',
            data=df_target,
            get_position='[lon, lat]',
            get_color='[0, 120, 0]',
            get_radius=Settings.groundzone,
            get_line_width=50,
            lineWidthMinPixels=2,
            stroked=True,
            filled=False,

        ),

        pdk.Layer(
            'ScatterplotLayer',
            data=df_target,
            get_position='[lon, lat]',
            get_color='[0, 0, 200]',
            get_radius=30,
        ),
    ]


def get_current_drone_layers(df_drone: pd.DataFrame, step: int) -> [pdk.Layer]:
    df_current = df_drone[df_drone.step == step]

    return [
        pdk.Layer(
            'ScatterplotLayer',
            data=df_current,
            get_position='[lon, lat, zed]',
            get_color='color',
            get_radius=50,

        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=df_current,
            get_position='[lon, lat]',
            get_color=[50, 50, 50, 50],
            get_radius=50,

        ),
    ]


def get_path_layers(df_path: pd.DataFrame, step: int) -> [pdk.Layer]:
    df_current = df_path[df_path.step == step]
    return [
        pdk.Layer(
            type="PathLayer",
            data=df_current,
            pickable=True,
            get_color="color",
            width_scale=10,
            width_min_pixels=1,
            get_path="path",
            get_width=1,
        )
    ]


# and ... do not forget
run(with_streamlit=True)
# run(blues=1, reds=3, with_streamlit=False)
