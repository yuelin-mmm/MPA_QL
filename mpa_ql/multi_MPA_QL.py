"""
MPA-QL for multi-UAV path planning.
"""

import math
import numpy as np

import CostFunction
import Initialization
import common
from MPA_QL import (
    diversity_centroid,
    encode_state,
    get_reward,
    is_contracted,
)

def _apply_de_update(prey, NP, uav_num, N_wp, dim, rb, P):
    # diff_mut: DE/rand/2 step
    r_num = np.array([np.random.permutation(NP) for _ in range(NP)])[:, :5]
    r_val = np.random.uniform(0, 1, (NP, uav_num, N_wp, dim))
    step_size = rb * (
        prey[r_num[:, 0], :, :, :]
        + np.random.uniform(0, 1, (NP, uav_num, N_wp, dim))
        * (prey[r_num[:, 1], :, :, :] - prey[r_num[:, 2], :, :, :])
        + np.random.uniform(0, 1, (NP, uav_num, N_wp, dim))
        * (prey[r_num[:, 3], :, :, :] - prey[r_num[:, 4], :, :, :])
    )
    prey[:, :, 1:-1, 1:] = prey[:, :, 1:-1, 1:] + P * r_val[:, :, 1:-1, 1:] * step_size[:, :, 1:-1, 1:]
    return prey

def _apply_dyn_update(prey, elite, rl, rb, cf_new, first_half_mask, P):
    # dyn_alt
    r_val = np.random.uniform(0, 1, size=prey.shape)
    # first half of the population
    step_size1 = rl * (elite - rl * prey)
    if np.random.random() < 0.5:
        # sin
        update1 = (
            P * r_val[:, :, 1:-1, 1:]
            * np.sin(2 * np.pi * np.random.uniform(0, 1))
            * step_size1[:, :, 1:-1, 1:]
        )
    else:
        # cos
        update1 = (
            P * r_val[:, :, 1:-1, 1:]
            * np.cos(2 * np.pi * np.random.uniform(0, 1))
            * step_size1[:, :, 1:-1, 1:]
        )

    # second half of the population
    step_size2 = rb * (rb * elite - prey)
    if np.random.random() < 0.5:
        # sin
        update2 = cf_new * np.sin(2 * np.pi * np.random.uniform(0, 1)) * step_size2[:, :, 1:-1, 1:]
    else:
        # cos
        update2 = cf_new * np.cos(2 * np.pi * np.random.uniform(0, 1)) * step_size2[:, :, 1:-1, 1:]
    prey[:, :, 1:-1, 1:] = np.where(
        first_half_mask,
        prey[:, :, 1:-1, 1:] + update1,
        elite[:, :, 1:-1, 1:] + update2,
    )
    return prey

def _apply_cobl_update(
    prey, pos_clip, pos_opposite_clip, elite, rl, cf_new, all_length,
    height_max, height_min, scenario_num, Q_coll, phi_max, psi_max,
    mu1, mu2, mu3, mu4, safe_dis, lb_array, ub_array, P):
    # cobl: candidate vs centroid opposition
    centroid_pop = np.mean(prey, axis=0)
    step_size = rl * (rl * elite - prey)
    pos_cobl = elite.copy()
    pos_opposite = prey.copy()
    pos_cobl[:, :, 1:-1, 1:] = elite[:, :, 1:-1, 1:] + P * cf_new * step_size[:, :, 1:-1, 1:]
    pos_opposite[:, :, 1:-1, 1:] = 2 * centroid_pop[:, 1:-1, 1:] - pos_cobl[:, :, 1:-1, 1:]
    pos_clip[:, :, 1:-1, :] = np.clip(pos_cobl[:, :, 1:-1, :], lb_array, ub_array)
    pos_opposite_clip[:, :, 1:-1, :] = np.clip(pos_opposite[:, :, 1:-1, :], lb_array, ub_array)
    return CostFunction.select_better_prey(
        pos_clip, pos_opposite_clip, all_length, height_max, height_min,
        scenario_num, Q_coll, phi_max, psi_max, mu1, mu2, mu3, mu4, safe_dis)

def multi_MPA_QL(
    NP, uav_num, N_wp, dim, P_start, P_target, lb_array, ub_array, Max_t,
    height_max, height_min, scenario_num, Q_coll, phi_max, psi_max,
    mu1, mu2, mu3, mu4, safe_dis, rho, H, tau, n_state, n_action, P, FADs,
    levy_scale, levy_alpha, eps, gamma, alpha0):
    # multi-UAV MPA-QL implementation
    w = rho * Max_t # stagnation window

    # top predator (best-so-far) and prey population
    top_predator_pos = np.zeros((uav_num, N_wp, dim))
    top_predator_fit = math.inf
    top_predator_fit_old = math.inf
    top_predator_length = np.linalg.norm(P_target - P_start)
    convergence_curve = []
    all_fitness = np.zeros(NP)
    all_length = top_predator_length

    # initialize prey population
    prey = Initialization.init_population(
        NP, uav_num, N_wp, dim, P_start, P_target, ub_array, lb_array
    )
    x_min = np.ones((NP, uav_num, N_wp - 2, dim)) * lb_array
    x_max = np.ones((NP, uav_num, N_wp - 2, dim)) * ub_array

    # Q-learning state
    q_table = np.zeros((n_state, n_action))
    curr_state = 0
    all_reward = []
    new_reward = 0.0
    g = 0
    d_history = []
    iter_idx = 0

    half = NP // 2
    first_half_mask = np.arange(NP) < half
    first_half_mask = np.expand_dims(first_half_mask, axis=(1, 2, 3))

    while iter_idx < Max_t:
        # evaluate population, update top predator
        prey[:, :, 1:-1, :] = np.clip(prey[:, :, 1:-1, :], lb_array, ub_array)
        all_fitness, path_length = CostFunction.cost_sum(
            prey, all_length, height_max, height_min, scenario_num, Q_coll,
            phi_max, psi_max, mu1, mu2, mu3, mu4, safe_dis)
        _, pos, fit, plen = common.population_best(prey, all_fitness, path_length)
        if fit < top_predator_fit:
            top_predator_fit = fit
            top_predator_pos = pos
            top_predator_length = plen

        elite = common.elite_matrix(top_predator_pos, NP)

        # memory: keep previous prey if it was better
        if iter_idx == 0:
            fitness_old = all_fitness.copy()
            prey_old = prey.copy()
        inx = fitness_old < all_fitness
        indx = np.tile(inx.reshape(-1, 1, 1, 1), (1, uav_num, N_wp, dim))
        prey = indx * prey_old + ~indx * prey
        all_fitness = inx * fitness_old + ~inx * all_fitness
        fitness_old = all_fitness.copy()
        prey_old = prey.copy()

        rl = levy_scale * common.levy((NP, uav_num, N_wp, dim), levy_alpha)  # Levy flight noise
        rb = np.random.randn(NP, uav_num, N_wp, dim)  # Brownian noise
        # cf_new: contraction factor
        cf_new = common.cf_new(iter_idx, Max_t)
        learning_rate = 1 - alpha0 * iter_idx / Max_t  # QL learning rate

        # epsilon-greedy action
        if np.random.uniform(0, 1) < eps:
            action = np.random.randint(0, n_action)
        else:
            action = np.argmax(q_table[curr_state, :])

        pos_clip = pos_opposite_clip = prey.copy()

        if action == 0:
            prey = _apply_de_update(prey, NP, uav_num, N_wp, dim, rb, P)
        elif action == 1:
            prey = _apply_dyn_update(prey, elite, rl, rb, cf_new, first_half_mask, P)
        else:
            prey = _apply_cobl_update(
                prey, pos_clip, pos_opposite_clip, elite, rl, cf_new, all_length,
                height_max, height_min, scenario_num, Q_coll, phi_max, psi_max,
                mu1, mu2, mu3, mu4, safe_dis, lb_array, ub_array, P)

        # re-evaluate after action update
        prey[:, :, 1:-1, :] = np.clip(prey[:, :, 1:-1, :], lb_array, ub_array)
        all_fitness, path_length = CostFunction.cost_sum(
            prey, all_length, height_max, height_min, scenario_num, Q_coll,
            phi_max, psi_max, mu1, mu2, mu3, mu4, safe_dis)
        _, pos, fit, plen = common.population_best(prey, all_fitness, path_length)
        if fit < top_predator_fit:
            top_predator_fit = fit
            top_predator_pos = pos
            top_predator_length = plen

        # memory again after action
        if iter_idx == 0:
            fitness_old = all_fitness.copy()
            prey_old = prey.copy()
        inx = fitness_old < all_fitness
        indx = np.tile(inx.reshape(-1, 1, 1, 1), (1, uav_num, N_wp, dim))
        prey = indx * prey_old + ~indx * prey
        all_fitness = inx * fitness_old + ~inx * all_fitness
        fitness_old = all_fitness.copy()
        prey_old = prey.copy()

        # FADs effect
        if np.random.uniform(0, 1) <= FADs:
            u_mask = np.random.rand(NP, uav_num, N_wp - 2, dim - 1) < FADs
            prey[:, :, 1:-1, 1:] = prey[:, :, 1:-1, 1:] + cf_new * (
                (
                    x_min[:, :, :, 1:]
                    + np.random.rand(NP, uav_num, N_wp - 2, dim - 1)
                    * (x_max[:, :, :, 1:] - x_min[:, :, :, 1:])
                )
                * u_mask
            )
        else:
            r_val = np.random.uniform(0, 1)
            prey[:, :, 1:-1, 1:] = prey[:, :, 1:-1, 1:] + (FADs * (1 - r_val) + r_val) * (
                prey[np.random.permutation(NP), :, 1:-1, 1:]
                - prey[np.random.permutation(NP), :, 1:-1, 1:]
            )

        # re-evaluate after FADs
        prey[:, :, 1:-1, :] = np.clip(prey[:, :, 1:-1, :], lb_array, ub_array)
        all_fitness, path_length = CostFunction.cost_sum(
            prey, all_length, height_max, height_min, scenario_num, Q_coll,
            phi_max, psi_max, mu1, mu2, mu3, mu4, safe_dis)
        _, pos, fit, plen = common.population_best(prey, all_fitness, path_length)
        if fit < top_predator_fit:
            top_predator_fit = fit
            top_predator_pos = pos
            top_predator_length = plen

        # stagnation counter g, reward, diversity state, Q update
        improved = top_predator_fit < top_predator_fit_old
        g = 0 if improved else g + 1
        reward = get_reward(top_predator_fit, top_predator_fit_old)
        new_reward += reward
        all_reward.append(new_reward)

        d_t = diversity_centroid(prey)
        contracted = is_contracted(d_t, d_history, tau, H)
        next_state = encode_state(g, contracted, w)
        d_history.append(d_t)

        # Q-learning update
        q_table[curr_state, action] += learning_rate * (
            reward + gamma * np.max(q_table[next_state, :]) - q_table[curr_state, action]
        )

        # update current state and top predator fitness
        curr_state = next_state
        top_predator_fit_old = top_predator_fit
        iter_idx += 1
        convergence_curve.append(top_predator_fit)

    return (
        top_predator_fit,
        convergence_curve,
        top_predator_pos,
        all_reward,
        q_table,
        top_predator_length,
    )
