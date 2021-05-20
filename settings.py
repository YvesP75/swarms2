import param_
import streamlit as st


class Settings:

    latlon: list = param_.LATLON['Paris']['lat'], param_.LATLON['Paris']['lon']

    perimeter: int = param_.PERIMETER
    perimeter_z: int = param_.PERIMETER_Z
    groundzone: int = param_.GROUNDZONE

    blues: int = param_.BLUES

    blues_per_circle: list = param_.BLUES_PER_CIRCLE
    blue_circles_rho: list = param_.BLUE_CIRCLES_RHO
    blue_circles_theta: list = param_.BLUE_CIRCLES_THETA
    blue_circles_zed: list = param_.BLUE_CIRCLES_ZED

    blue_speed_init: int = param_.BLUE_SPEED_INIT

    reds: int = param_.REDS

    red_squads: list = param_.RED_SQUADS
    red_squads_rho: list = param_.RED_SQUADS_RHO
    red_squads_theta: list = param_.RED_SQUADS_THETA
    red_squads_zed: list = param_.RED_SQUADS_ZED

    red_rho_noise: list = param_.RED_RHO_NOISE
    red_theta_noise: list = param_.RED_THETA_NOISE
    red_zed_noise: list = param_.RED_ZED_NOISE

    red_speed_init: int = param_.RED_SPEED_INIT

    def define(with_streamlit: bool = True):
        """"
            shows the blue and red swarms in Streamlit
            :return:
            """
        blues = Settings.blues
        reds = Settings.reds

        if with_streamlit:
            st.title('Blue Swarm against Red Swarm by aikos')
            st.write('This is a quick demo controlled by Reinforcement Learning.')
            st.text('<- Set parameters')

            st.sidebar.subheader("Define the battlefield")
            blues = st.sidebar.slider("how many blues to defend the sweet pot?", 1, 12)
            Settings.blues = blues
            reds = st.sidebar.slider("how many reds are on the attack?", 1, 12)
            Settings.blues = reds
            location = st.sidebar.radio("Location", ['Paris', 'Puilaurens', 'San Francisco'])

            lat_tg = param_.LATLON[location]['lat']
            lon_tg = param_.LATLON[location]['lon']

            Settings.latlon = lat_tg, lon_tg

            st.sidebar.write(
                'you probably need more drones '
                'No worries, we have plenty at www.aikos.com ')


        return blues, reds
