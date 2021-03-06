from dataclasses import dataclass
import numpy as np
from stable_baselines3 import SAC
from os import path

import param_
from drone import Drone


@dataclass
class SwarmPolicy:
    blues: int
    reds: int
    is_blue: bool
    model: object = None
    count: int = 0

    def __post_init__(self):

        dir_path = "policies/last" + f"/b{self.blues}r{self.reds}/"
        model_path = dir_path + ("blues_last.zip" if self.is_blue else "reds_last.zip")
        if path.exists(model_path):
            print("model loaded:" + model_path)
            self.model = SAC.load(model_path, verbose=1)

    # predicts from the model or from a simple centripete model
    def predict(self, obs):

        self.count += 1

        if self.model:
            action, _ = self.model.predict(obs)
            # verbose = 'prediction from ' + (' blue model' if self.is_blue else ' red model') + ' at ' + str(self.count)
            # print(verbose)
            return action
        else:
            if self.is_blue:
                return self._improved_attack_predict(obs)
            else:
                return self._simple_predict(obs)

    # the default policy
    def _simple_predict(self, obs):
        simple_obs = _decentralise(obs[0:self.blues*6] if self.is_blue else obs[self.blues*6:(self.blues+self.reds)*6])
        drone = Drone(is_blue=self.is_blue)
        action = np.array([])
        nb_drones = self.blues if self.is_blue else self.reds
        for d in range(nb_drones):
            assign_pos_speed(drone, d, simple_obs)
            '''
            pos_n, speed_n = simple_obs[d*6:d*6+3], simple_obs[d*6+3:d*6+6]
            pos = drone.from_norm(pos_n, drone.max_positions, drone.min_positions)
            drone.position = pos
            speed = drone.from_norm(speed_n, drone.max_speeds, drone.min_speeds)
            drone.speed = speed
            '''
            action_d = drone.simple_red()
            action = np.hstack((action, action_d))

        action = _centralise(action)
        return action


    # the default attack policy
    def _attack_predict(self, obs):

        def assign_targets(friends_obs, foes_obs):
            '''
            this current version is simplistic: all friends target the first foe :)
            :param obs:
            :return:
            '''
            friends_nb = len(friends_obs) // 6
            foes_nb = len(foes_obs) // 6

            friends_targets = -np.ones(friends_nb, dtype=int)
            while -1 in friends_targets:
                for foe in range(foes_nb):
                    foe_pos = _denorm(foes_obs[foe*6:foe*6+3])
                    foe_pos_z = foe_pos[0] * np.exp(1j * foe_pos[1])
                    min_distance = np.inf
                    closest_friend = -1
                    for friend in range(friends_nb):
                        if friends_targets[friend] == -1:
                            friend_pos = _denorm(friends_obs[friend*6:friend*6+3])
                            friend_pos_z = friend_pos[0] * np.exp(1j * friend_pos[1])
                            distance = np.abs(foe_pos_z - friend_pos_z) ** 2 + (friend_pos[2] - foe_pos[2]) ** 2
                            if distance < min_distance:
                                min_distance = distance
                                closest_friend = friend
                    friends_targets[closest_friend] = foe

            return friends_targets


        # gets the friends and foes obs
        blue_obs = _decentralise(obs[0:self.blues * 6])
        red_obs = _decentralise(obs[self.blues * 6:(self.blues + self.reds) * 6])
        friends_obs = blue_obs if self.is_blue else red_obs
        foes_obs = red_obs if self.is_blue else blue_obs

        # assign red targets to blues
        friends_targets = assign_targets(friends_obs, foes_obs)

        friend_drone = Drone(is_blue=self.is_blue)
        foe_drone = Drone(is_blue=not self.is_blue)

        action = np.array([])
        nb_drones = self.blues if self.is_blue else self.reds
        for d in range(nb_drones):

            # assign denormalised position and speed (in m and m/s) to foe drone
            friend_drone = assign_pos_speed(friend_drone, d, friends_obs)
            foe_drone_id = friends_targets[d]

            foe_drone = assign_pos_speed(foe_drone, foe_drone_id, foes_obs)
            target, time_to_target = calculate_target(friend_drone, foe_drone)
            action_d = friend_drone.simple_red(target=target, z_margin=0)
            action = np.hstack((action, action_d))

        action = _centralise(action)
        return action

        # the improved manual attack policy
    def _improved_attack_predict(self, obs):
        # TODO: revamp the algo as follows
        #  start from closest reds, find all blues that are compatible with some margin
        #  among those blues, choose the blue whose first target is the latest
        #  until there is no red left
        #  in case there are blues left overs, restart the process, or converge to zero, or..
        #  or we decide in advance how many blues we want on the closest and populate several blues againts reds
        #  at the beginning
        # TODO: check that reds are correctly ordered
        # TODO : add margin in the params
        # TODO : case of the foe is not reachable

        # gets the friends and foes obs
        blue_obs = _decentralise(obs[0:self.blues * 6])
        red_obs = _decentralise(obs[self.blues * 6:(self.blues + self.reds) * 6])
        friends_obs = blue_obs if self.is_blue else red_obs
        foes_obs = red_obs if self.is_blue else blue_obs

        friends_nb = self.blues if self.is_blue else self.reds
        foes_nb = self.reds if self.is_blue else self.blues

        friend_drones = []
        for friend_id in range(friends_nb):
            # assign denormalised position and speed (in m and m/s) to foe drone
            friend_drone = Drone(is_blue=self.is_blue)
            friend_drone = assign_pos_speed(friend_drone, friend_id, friends_obs)
            friend_drones.append(friend_drone)

        foe_drones = []
        for foe_id in range(foes_nb):
            # assign denormalised position and speed (in m and m/s) to foe drone
            foe_drone = Drone(is_blue=not self.is_blue)
            foe_drone = assign_pos_speed(foe_drone, foe_id, foes_obs)
            foe_drones.append(foe_drone)

        targets = np.zeros((friends_nb, foes_nb, 3))
        best_targets = -np.ones((friends_nb, 3))
        times_to_target = -np.ones((friends_nb, foes_nb))
        calculation_done = -np.ones(friends_nb)
        friend_chosen = -np.ones(friends_nb)

        foe_id = 0
        friends_chosen = 0
        while foe_id < foes_nb-1 and friends_chosen < friends_nb:
            best_friend = -1
            best_target = np.zeros(3)
            longest_time = -np.inf
            foe_drone = foe_drones[foe_id]
            for friend_id in range(friends_nb):
                if friend_chosen[friend_id] == -1:  # the friend has no foe target assigned

                    friend_drone = friend_drones[friend_id]

                    if calculation_done[friend_id] == -1:  # it has not already been calculated
                        target_, time_to_target, is_a_catch = calculate_target(friend_drone, foe_drone)
                        times_to_target[friend_id][foe_id] = time_to_target if is_a_catch else np.inf
                        targets[friend_id][foe_id] = target_
                    if times_to_target[friend_id][foe_id] < np.inf:  # it is a catch

                        if calculation_done[friend_id] == -1:  # calculation of time with other drones has not been done
                            for foe_idx in range(foe_id + 1, foes_nb):
                                foex_drone = foe_drones[foe_idx]
                                target_, time_to_target, is_a_catch = calculate_target(friend_drone, foex_drone)
                                times_to_target[friend_id][foe_idx] = time_to_target if is_a_catch else np.inf
                                targets[friend_id][foe_idx] = target_
                                calculation_done[friend_id] = 1
                        closest_target = np.min(times_to_target[friend_id, foe_id+1:])
                        if longest_time < closest_target:
                            longest_time = closest_target
                            best_friend = friend_id
                            best_target = targets[friend_id][foe_id]

            best_targets[best_friend] = best_target
            friend_chosen[best_friend] = foe_id
            friends_chosen += 1
            foe_id += 1

        if friends_chosen < friends_nb:
            last_foe = foes_nb - 1
            for friend_id in range(friends_nb):
                if friend_chosen[friend_id] == -1:
                    if times_to_target[friend_id, last_foe] == -1:
                        friend_drone, foe_drone = friend_drones[friend_id], foe_drones[last_foe]
                        target_, time_to_target, is_a_catch = calculate_target(friend_drone, foe_drone)
                        targets[friend_id][last_foe] = target_
                    closest_target_id = np.argmin(times_to_target[friend_id, :])
                    best_targets[friend_id] = targets[friend_id][closest_target_id]




        action = np.array([])
        for friend_id in range(friends_nb):
            action_d = friend_drones[friend_id].simple_red(target=best_targets[friend_id], z_margin=0)
            action = np.hstack((action, action_d))

        action = _centralise(action)
        return action


def assign_pos_speed(drone: Drone, d: int, obs: np.ndarray) -> Drone:
    # assign denormalised position and speed (in m and m/s) to friend drone
    d = int(d)
    pos_n, speed_n = obs[d*6:d*6+3], obs[d*6+3:d*6+6]
    pos = drone.from_norm(pos_n, drone.max_positions, drone.min_positions)
    drone.position = pos
    speed = drone.from_norm(speed_n, drone.max_speeds, drone.min_speeds)
    drone.speed = speed
    return drone


def _denorm(pos):  # from norm (i.e. already decentralised) to meter
    drone = Drone()
    pos_meter = drone.from_norm(pos, drone.max_positions, drone.min_positions)
    return pos_meter


def _decentralise(obs):  # [-1,1] to [0,1]
    obs = (obs+1)/2
    return obs


def _centralise(act):  # [0,1] to [-1,1]
    act = (act - 1/2) * 2
    return act


def calculate_target(blue_drone: Drone, red_drone: Drone) -> (np.ndarray(3, ), float, bool):
    '''

    :param blue_drone:
    :param red_drone:
    :return:
    '''
    # TODO : be more precise at the end of the discovery process

    def transform(pos, delta_, theta_):
        pos[0] -= delta_
        pos[1] -= theta_
        return pos[0] * np.exp(1j * pos[1])

    def untransform_to_array(pos, delta_, theta_):
        pos[0] += delta_
        pos[1] += theta_
        return pos

    theta = red_drone.position[1]
    delta = param_.GROUNDZONE

    attack_pos = np.copy(blue_drone.position)
    target_pos = np.copy(red_drone.position)

    z_blue = transform(attack_pos, delta, theta)
    z_red = np.real(transform(target_pos, delta, theta))

    v_blue = blue_drone.drone_model.max_speed
    v_red = red_drone.drone_model.max_speed

    blue_shooting_distance = blue_drone.drone_model.distance_to_neutralisation

    blue_time_to_zero = (np.abs(z_blue) - blue_shooting_distance) / v_blue
    red_time_to_zero = z_red / v_red

    if red_time_to_zero <= param_.STEP or red_time_to_zero < blue_time_to_zero + param_.STEP:
        return np.zeros(3), red_time_to_zero, False
    else:
        max_target = z_red
        min_target = 0
        while True:
            target = (max_target + min_target) / 2
            blue_time_to_target = max(0, (np.abs(z_blue - target) - blue_shooting_distance) / v_blue)
            red_time_to_target = np.abs(z_red - target) / v_red
            if red_time_to_target - param_.STEP < blue_time_to_target <= red_time_to_target:
                target = untransform_to_array((target / z_red) * target_pos, delta, theta)
                return target, blue_time_to_target, True
            if red_time_to_target < blue_time_to_target:
                max_target = target
                min_target = min_target
            else:  # blue_  time_to_target  <= red_time_to_target -1:
                max_target = max_target
                min_target = target




def unitary_test(rho_blue: float, theta_blue: float, rho_red: float, theta_red: float):
    '''
    tests for the calculate target function
    :param rho_blue:
    :param theta_blue:
    :param rho_red:
    :param theta_red:
    :return:
    '''
    blue_drone = Drone()
    blue_drone.position = np.array([rho_blue, theta_blue, 100])
    red_drone = Drone(is_blue=False)
    red_drone.position = np.array([rho_red, theta_red, 100])
    tg, time = calculate_target(blue_drone, red_drone)
    print('rho_blue : ', rho_blue, ' theta_blue : ', theta_blue, ' rho_red : ', rho_red, ' theta_red : ', theta_red,
          ' tg : ', tg, ' time : ', time)
    return tg, time


def test():
    '''
    test for the calculate trajectory function
    :return:
    '''
    for rho_blue in [1000]:
        for theta_blue in np.pi * np.array([-1, 0.75, 0.5, 0.25, 0]):
            for rho_red in [1000]:
                for theta_red in np.pi * np.array([0, 1/4]):
                    unitary_test(rho_blue=rho_blue, theta_blue=theta_blue, rho_red=rho_red, theta_red=theta_red)
    print('done')



