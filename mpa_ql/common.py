"""
Shared MPA functions for CEC and multi-UAV.
"""

import math
import numpy as np

def levy(shape, alpha):
    # Levy flight samples
    # shape: (NP, dim) or (NP, uav, N_wp, dim)
    sigmax_num = math.gamma(1 + alpha) * math.sin(math.pi * alpha / 2)
    sigmax_den = math.gamma((1 + alpha) / 2) * alpha * (2 ** ((alpha - 1) / 2))
    sigmax = (sigmax_num / sigmax_den) ** (1 / alpha)
    x = np.random.normal(0, sigmax, shape)
    y = np.random.normal(0, 1, shape)
    return x / (np.abs(y) ** (1 / alpha))

def cf_new(iter_idx, Max_t):
    # MPA-QL control factor CF_new
    return (np.sin((np.pi / 2) * (1 - iter_idx / Max_t))) ** (2 * iter_idx / Max_t)

def elite_matrix(top_pos, NP):
    # broadcast top predator to (NP, ...) elite matrix
    return np.broadcast_to(top_pos, (NP,) + top_pos.shape)

def population_best(prey, all_fitness, path_length):
    # argmin over population: idx, position, fitness, path length
    best_idx = np.argmin(all_fitness)
    return best_idx, prey[best_idx].copy(), all_fitness[best_idx], path_length[best_idx]
