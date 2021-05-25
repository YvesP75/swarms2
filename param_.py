
import numpy as np

STEP = 1  # seconds per time step
DURATION = 1000  # seconds


STEP_COST = 1/1000
HEURISTIC_WEIGHT = 1/100
RED_COST = 1/10
OOB_COST = 1

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


BLUE_SPEED_INIT = 1  # in ratio to max_speed

BLUE_COLOR = [0, 0, 150, 120]
BLUE_DEAD_COLOR = [20, 20, 60]

# red team init

REDS = 12

RED_SQUADS = [3, 3, 3, 3, 3, 3]
RED_SQUADS_RHO = [700, 1000, 1400, 1800, 2200, 2600]
RED_SQUADS_THETA = np.pi * np.array([0, 1/4, -1/4, -1/2, 1/2, 0])
RED_SQUADS_ZED = [300, 250, 100, 250, 200, 100]
RED_DISTANCE_FACTOR = 1

RED_RHO_NOISE = [50, 50, 50, 200, 200, 300]
RED_THETA_NOISE = np.pi * np.array([1/4, 1/4, 1/4, 1/4, 1/4, 1/4])
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
          'duration_to_neutralisation': 2,
          'Cxy': 0.3,  # horizontal air resistance : link to speed max by the relation Fxy_max = Cxy * Speedxy_max
          'Cz': 0.8,  # vertical air resistance : link to speed max by the relation Fz_max = Cz * Speedz_max
          'mass': 40,  # kg
          'Fz_min_ratio': 0.5,  # how much weight is compensated (below 1 => drone goes down)
          'Fz_max_ratio': 1.8,  # how much weight is compensated (>1 => drone goes up)
          'Fxy_ratio': 0.6,  # Force xy relative to weight
     },
}
