"""
works with all the transformations and calculation associated to position, speed, acceleration
"""

import numpy as np


def position_to_xyz(position: [float]) -> [float]:
    """
    allows to get the 3D xyz coordinates from a polar representation
    :param position: array (3,) with rho in meter, theta in rad, zed in meter
    :return: float array (3,) with x, y, z in meter
    """
    pos = position[0] * np.exp(1j * position[1])
    return [np.real(pos), np.imag(pos), position[2]]


"""
def _test_position_to_norm():
    assert position_to_norm([param.PERIMETER, 0, 100]) == [1, 0, 1]
    assert position_to_norm([0, -np.pi / 2, 0]) == [0, 0.75, 0]
    assert position_to_norm([0, np.pi / 2, 0]) == [0, 0.25, 0]
"""


def is_in_the_cone(position1: [float], position2: [float], vector2: [float], angle: float) -> bool:
    """
    checks if the point @ position 2 is in the cone from position 1 with an angle of angle
    :param position1: in x, y, z
    :param position2: in x, y, z
    :param vector2: in x, y, z
    :param angle: in rad
    :return:
    """
    vector1 = np.array(position2, dtype=float) - np.array(position1)
    vector1 /= np.linalg.norm(vector1)
    vector2 = np.array(vector2, dtype=float)
    vector2 /= np.linalg.norm(vector2)
    cos_theta = np.dot(vector1, vector2)
    if 0 < cos_theta:
        theta = np.arcsin(np.sqrt(1 - cos_theta ** 2))
        return theta < angle
    return False


def _test_is_in_the_cone():
    assert is_in_the_cone([0, 0, 0], [1, 0.1, 0], [1, 0, 0], np.pi / 5)
    assert is_in_the_cone([0, 0, 0], [1, 0.1, 0], [0, 1, 0], np.pi / 5)
    pass


def rhotheta_to_latlon(rho: float, theta: float, lat_tg: float, lon_tg: float) -> [float, float]:
    """
    transforms polar coordinates into lat, lon
    :param rho:
    :param theta:
    :param lat_tg: latitude de la target (0,0)
    :param lon_tg: longitude de la target (0,0)
    :return:
    """
    z = rho * np.exp(1j * theta)
    lat = np.imag(z) * 360 / (40075 * 1000) + lat_tg
    lon = np.real(z) * 360 / (40075 * 1000 * np.cos(np.pi / 180 * lat)) + lon_tg
    return lat, lon

