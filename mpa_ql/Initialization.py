"""
Population initialization for multi-UAV path planning.
"""

import numpy as np

def init_population(NP, uav_num, N_wp, dim, P_start, \
    P_target, ub_array, lb_array):
    prey = np.zeros((NP, uav_num, N_wp, dim))
    # fix start and end waypoints
    prey[:, :, 0, :] = P_start
    prey[:, :, -1, :] = P_target

    # x of interior waypoints along ST
    dist_st = P_target[0] - P_start[0]
    delta_x = dist_st / (N_wp - 1)
    x_coords = delta_x * np.arange(1, N_wp - 1)
    x_grid = np.tile(x_coords.reshape(-1, 1, 1), (1, uav_num, NP))
    prey[:, :, 1:-1, 0] = np.transpose(x_grid, (2, 1, 0))

    # y, z of interior waypoints
    random_vals = np.random.rand(NP, uav_num, N_wp - 2, dim - 1)
    prey[:, :, 1:-1, 1:] = random_vals * (ub_array[1:] - lb_array[1:]) + lb_array[1:]
    return prey
