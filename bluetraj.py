import numpy as np
import param_
from drone import Drone


def calculate_target(blue_drone: Drone, red_drone: Drone) -> np.ndarray(3, ):
    '''

    :param blue_drone:
    :param red_drone:
    :return:
    '''

    def transform(pos, delta, theta):
        pos[0] -= delta
        pos[1] -= theta
        return pos[0] * np.exp(1j * pos[1])

    def untransform_to_array(pos, delta, theta):
        pos[0] += delta
        pos[1] += theta
        return pos

    theta = red_drone.position[1]
    delta = param_.GROUNDZONE

    z_blue = transform(blue_drone.position, delta, theta)
    z_red = np.real(transform(red_drone.position, delta, theta))

    v_blue = blue_drone.drone_model.max_speed
    v_red = red_drone.drone_model.max_speed

    blue_shooting_distance = blue_drone.drone_model.distance_to_neutralisation

    blue_time_to_zero = (np.abs(z_blue) - blue_shooting_distance) / v_blue
    red_time_to_zero = z_red / v_red

    if red_time_to_zero <= param_.STEP or red_time_to_zero < blue_time_to_zero + param_.STEP:
        return np.zeros(3), red_time_to_zero
    else:
        max_target = z_red
        min_target = 0
        while True:
            target = (max_target + min_target) / 2
            blue_time_to_target = max(0, (np.abs(z_blue - target) - blue_shooting_distance) / v_blue)
            red_time_to_target = np.abs(z_red - target) / v_red
            if red_time_to_target - param_.STEP < blue_time_to_target <= red_time_to_target:
                target = untransform_to_array((target / z_red) * red_drone.position, delta, theta)
                return target, blue_time_to_target
            if red_time_to_target < blue_time_to_target:
                max_target = target
                min_target = min_target
            else:  # blue_  time_to_target  <= red_time_to_target -1:
                max_target = max_target
                min_target = target




def unitary_test(rho_blue: float, theta_blue: float, rho_red: float, theta_red: float):
    blue_drone = Drone()
    blue_drone.position = np.array([rho_blue, theta_blue, 100])
    red_drone = Drone(is_blue=False)
    red_drone.position = np.array([rho_red, theta_red, 100])
    tg, time = calculate_target(blue_drone, red_drone)
    print('rho_blue : ', rho_blue, ' theta_blue : ', theta_blue, ' rho_red : ', rho_red, ' theta_red : ', theta_red,
          ' tg : ', tg, ' time : ', time)
    return tg, time





def test():
    for rho_blue in [1000]:
        for theta_blue in np.pi * np.array([-1, 0.75, 0.5, 0.25, 0]):
            for rho_red in [1000]:
                for theta_red in np.pi * np.array([0, 1/4]):
                    unitary_test(rho_blue=rho_blue, theta_blue=theta_blue, rho_red=rho_red, theta_red=theta_red)
    print('done')

