
import numpy as np

STEP = 1  # seconds per time step
DURATION = 200  # seconds

POLICY_FOLDER = 'default_policies'

STEP_COST = 1/100
OOB_COST = 10  # Out Of Bound : when the drone is below 0 or above a PERIMETER_Z
WIN_REWARD = 10  # either Blues or Reds have won
RED_SHOT_REWARD = 1/10  # when a red drone is shot
TARGET_HIT_COST = 1/10  # when a red drone hits the target
THREAT_WEIGHT = 1/100  # when reds are close to the target (* function of the red distance)
TTL_COST = 5  # when a red is still alive after its TTL: it is a failure for both blues and reds
TTL_RATIO = 3  # margin for red drones to get to the target if they went full speed

ELEVATION_SCALE = 1
TRAJ_LENGTH = 6
SIMU_SPEED = 0.2


"""
the playground parameters
"""

PERIMETER = 5000
PERIMETER_Z = 600

# PERIMETER of the ground zone to defend
GROUNDZONE = 100

# position in LATLON
LATLON = {'Paris':
              {'lat': 48.865879, 'lon': 2.319827},
          'Fonsorbes':
              {'lat': 43.54, 'lon': 1.25},
          'San Francisco':
              {'lat': 37.7737283, 'lon': -122.4342383},
          'Puilaurens':
              {'lat': 42.803943093860894, 'lon': 2.299540897567384},
          }

"""
the Team Parameters
"""

# blue team init

BLUES = 12

BLUES_PER_CIRCLE = [4, 4, 4, 4, 4, 4]
BLUE_CIRCLES_RHO = [500, 900, 1400, 1600, 2000, 2500]
BLUE_CIRCLES_THETA = [0, -np.pi/3, -np.pi, -np.pi/2, 0, np.pi/3]
BLUE_CIRCLES_ZED = [300, 250, 250, 100, 250, 100]
BLUE_DISTANCE_FACTOR = 1

BLUE_IS_UNKILLABLE = True


BLUE_SPEED_INIT = 1  # in ratio to max_speed

BLUE_COLOR = [0, 0, 150, 120]
BLUE_DEAD_COLOR = [20, 20, 60]

# red team init

REDS = 12

RED_SQUADS = [3, 3, 3, 3, 3, 3]
RED_SQUADS_RHO = [300, 700, 1000, 1200, 1500, 2000]
RED_SQUADS_THETA = np.pi * np.array([-1/2, 1/4, -1/4, -1/2, 1/2, 0])
RED_SQUADS_ZED = [300, 250, 100, 250, 200, 100]
RED_DISTANCE_FACTOR = 1


RED_RHO_NOISE = [200, 300, 200, 200, 200, 300]
RED_THETA_NOISE = np.pi * np.array([2, 2, 1, 1, 1, 1])
RED_ZED_NOISE = [20, 50, 10, 10, 50, 60]

RED_SPEED_INIT = 0.0  # in ratio to max_speed

RED_COLOR = [150, 0, 0, 120]
RED_DEAD_COLOR = [120, 50, 30]
RED_SUCCESS_COLOR = [200, 200, 0]
BLACK_COLOR = [0, 0, 0]
GREEN_COLOR = [0, 255, 255]

"""
the Drone Parameters
"""

g = 9.81

DRONE_MODEL = ['beta', 'alpha']  # blue = DRONE_MODEl[1]

DRONE_MODELS = {
    'alpha': {
          'angle_to_neutralisation': np.pi / 4,  # in rad
          'distance_to_neutralisation': 250,  # in m
          'duration_to_neutralisation': 2,  # in s
          'Cxy': 0.2,  # horizontal air resistance  = Cxy * v^2
          'Cz': 0.7,  # vertical air resistance
          'mass': 50,  # kg
          'Fz_min_ratio': 0.6,  # how much weight is compensated (below 1 => drone goes down)
          'Fz_max_ratio': 1.4,  # how much weight is compensated (>1 => drone goes up)
          'Fxy_ratio': 1,  # Force xy relative to weight
    },
     'beta': {
          'angle_to_neutralisation': np.pi / 4,
          'distance_to_neutralisation': 250,
          'duration_to_neutralisation': np.inf,
          'Cxy': 0.3,  # horizontal air resistance : link to speed max by the relation Fxy_max = Cxy * Speedxy_max
          'Cz': 0.8,  # vertical air resistance : link to speed max by the relation Fz_max = Cz * Speedz_max
          'mass': 40,  # kg
          'Fz_min_ratio': 0.5,  # how much weight is compensated (below 1 => drone goes down)
          'Fz_max_ratio': 1.8,  # how much weight is compensated (>1 => drone goes up)
          'Fxy_ratio': 0.6,  # Force xy relative to weight
     },
}
