import numpy as np
from dataclasses import dataclass

import param_
from settings import Settings
from drone import Drone


@dataclass
class Playground:
    """
    This is a cylindrical 3D-env where blue drones defend a central zone from the attack of red drones
    the playground manages also the interactions between foe-drones such as the firing
    """

    perimeter = Settings.perimeter
    perimeter_z = Settings.perimeter_z
    groundzone = Settings.groundzone

    env: object
    blue_drones: [Drone]
    red_drones: [Drone]

    def __post_init__(self):
        # creates the fire matrices
        self.blues_have_fired_reds = np.zeros(shape=(len(self.blue_drones),
                                                     len(self.red_drones)), dtype=int)
        self.reds_have_fired_blues = np.zeros(shape=(len(self.red_drones),
                                                     len(self.blue_drones)), dtype=int)

        # how long the drone needs to have the other in target
        self.blue_shots_to_kill = param_.DRONE_MODELS[param_.DRONE_MODEL[True]]['duration_to_neutralisation']
        self.red_shots_to_kill = param_.DRONE_MODELS[param_.DRONE_MODEL[False]]['duration_to_neutralisation']
        self.blue_shots_to_kill //= param_.STEP
        self.red_shots_to_kill //= param_.STEP

        # how far can a drone shoot
        self.distance_blue_shot = param_.DRONE_MODELS[param_.DRONE_MODEL[True]]['distance_to_neutralisation']
        self.distance_red_shot = param_.DRONE_MODELS[param_.DRONE_MODEL[False]]['distance_to_neutralisation']


    def reset(self):
        self.blues_have_fired_reds[...] = 0
        self.reds_have_fired_blues[...] = 0

    def get_observation(self):
        return self.blues_have_fired_reds / self.blue_shots_to_kill, \
               self.reds_have_fired_blues / self.red_shots_to_kill

    def step(self):
        """
        determines who has fired who, and who is dead in the end
        :return: Tuple with list of Blue and Reds dead. (if a blue or a red is dead, the sequence is over)
        """
        # gets who has fired who in this step
        blues_fire_reds = np.array([[blue.fires_(red) for red in self.red_drones] for blue in self.blue_drones])
        reds_fire_blues = np.array([[red.fires_(blue) for blue in self.blue_drones] for red in self.red_drones])

        if 0 < np.sum(blues_fire_reds) + np.sum(reds_fire_blues):
            alpha = 0

        # if the foe is no longer seen, the count restarts from 0
        self.blues_have_fired_reds *= blues_fire_reds
        self.reds_have_fired_blues *= reds_fire_blues

        # and the count is incremented for the others
        self.blues_have_fired_reds += blues_fire_reds
        self.reds_have_fired_blues += reds_fire_blues

        # np magic : first find the list of duos shooter/shot, keep the shots (only once)
        red_deads = np.unique(np.argwhere(self.blues_have_fired_reds >= self.blue_shots_to_kill).T[1])
        blue_deads = np.unique(np.argwhere(self.reds_have_fired_blues >= self.red_shots_to_kill).T[1])

        # tell the drones that they are dead
        for drone_id in blue_deads:
            self.blue_drones[drone_id].is_killed(is_blue=True)
        for drone_id in red_deads:
            self.red_drones[drone_id].is_killed(is_blue=False)

        bf_obs, rf_obs = self.get_observation()
        bf_reward = rf_reward = 0
        bf_done, rf_done = len(red_deads), len(blue_deads)
        bf_info, rf_info = red_deads, blue_deads

        if bf_done + rf_done > 0:
            print('someone is killed: {0} blues and {1} reds'.format(rf_done, bf_done))

        return bf_obs, bf_reward, bf_done, bf_info, rf_obs, rf_reward, rf_done, rf_info

    def heuristic(self):

        # consider only living drones
        blue_drones = [drone for drone in self.blue_drones if drone.is_alive]
        red_drones = [drone for drone in self.blue_drones if drone.is_alive]

        # check that there still are some drones alive
        if len(blue_drones) * len(red_drones) == 0:
            print('fight is over : there are still {0} blues and {1} reds'.format(len(blue_drones), len(red_drones)))
            return 0, 0

        # create a matrix that gives for each blue drone the distance to each red drone
        rb_distance = np.array([[blue.distance(red) for red in red_drones] for blue in blue_drones])

        # the distance goes through a "norm" function to give a weight of 1 to the the closest drones
        normalised_rb_distance = np.exp(-0.5 * (rb_distance / self.distance_blue_shot) ** 2)

        # each blue drone contributes a % of its capacity on pressure to red. this % is such that sum of blue weight = 1
        sum_blue_weight = np.tile(np.sum(normalised_rb_distance, axis=1), (len(red_drones), 1)).T
        blue_weight = normalised_rb_distance / sum_blue_weight

        # the total blue pressure is added to get to pressure on each red
        avg_blue_weight = np.sum(blue_weight, axis=0)

        # this total goes through a log in order to render that the marginal weight of blue drones is decreasing
        blue_weight_on_red = np.tanh(avg_blue_weight)

        # distance of red drones to 0
        r_distance = np.array([red.distance() for red in red_drones])

        # distance of red drones to the groundzone (a priori, it should be impossible to have a negative r_distance)
        r_distance -= Settings.groundzone
        r_distance = np.clip(r_distance, 0, np.inf)  # just in case

        red_loss = np.exp(-0.5 * (r_distance / self.distance_blue_shot) ** 2)

        red_loss_weighted = red_loss * (1 - blue_weight_on_red)

        total_red_loss = np.sum(red_loss)
        total_red_loss_weighted = np.sum(red_loss_weighted)

        return total_red_loss, total_red_loss_weighted






