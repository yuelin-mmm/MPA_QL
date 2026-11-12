"""
MPA-QL for CEC continuous optimization.
"""

import math
import random
import numpy as np
import common

def diversity_centroid(pop):
    # mean distance of individuals to population centroid
    flat = pop.reshape(pop.shape[0], -1)
    centroid = flat.mean(axis=0)
    return np.linalg.norm(flat - centroid, axis=1).mean()

def is_contracted(d_t, d_history, tau, H):
    # True if D_t <= tau * mean(last H), history excludes current d_t
    if len(d_history) < H:
        return False
    d_ref = np.mean(d_history[-H:])
    return d_t <= tau * d_ref

def encode_state(g, contracted, w):
    # s0..s3 from (g > W_stag) x contracted
    return int(g > w) * 2 + int(contracted)

def get_reward(best_fit, best_fit_old):
    # +1 if global best improves, else -1
    return 1.0 if best_fit < best_fit_old else -1.0

def transform_boundaries(lb, ub, dim):
    # scalar bounds -> per-dimension arrays
    return np.full(dim, lb, dtype=float), np.full(dim, ub, dtype=float)

def random_pop(NP, dim, lb_array, ub_array):
    # uniform random population in [lb, ub]
    return np.random.rand(NP, dim) * (ub_array - lb_array) + lb_array

def MPA_QL(fobj, NP, Max_t, lb, ub, dim, rho, H, tau, \
    n_state, n_action, P, FADs, levy_scale, levy_alpha, eps, gamma, alpha0):
    # MPA-QL implementation
    w = rho * Max_t # stagnation window

    # top predator (best-so-far) and prey population
    top_predator_pos = np.zeros(dim)
    top_predator_fit = math.inf
    top_predator_fit_old = math.inf
    convergence_curve = []
    stepsize = np.zeros((NP, dim))
    all_fitness = np.full(NP, np.inf)

    lb_array, ub_array = transform_boundaries(lb, ub, dim)
    prey = random_pop(NP, dim, lb_array, ub_array)
    xmin = np.ones((NP, dim)) * lb
    xmax = np.ones((NP, dim)) * ub

    # Q-learning state
    q_table = np.zeros((n_state, n_action))
    curr_state = 0
    all_reward = []
    new_reward = 0.0
    g = 0
    d_history = []
    iter_idx = 0

    while iter_idx < Max_t:
        # evaluate population, update top predator
        for i in range(NP):
            prey[i] = np.clip(prey[i], lb, ub)
            all_fitness[i] = fobj(prey[i])
            if all_fitness[i] < top_predator_fit:
                top_predator_fit = all_fitness[i]
                top_predator_pos = prey[i].copy()

        elite = common.elite_matrix(top_predator_pos, NP)

        # memory: keep previous prey if it was better
        if iter_idx == 0:
            fitness_old = all_fitness.copy()
            prey_old = prey.copy()
        inx = fitness_old < all_fitness
        indx = np.tile(inx, (dim, 1)).T
        prey = indx * prey_old + ~indx * prey
        all_fitness = inx * fitness_old + ~inx * all_fitness
        fitness_old = all_fitness.copy()
        prey_old = prey.copy()

        rl = levy_scale * common.levy((NP, dim), levy_alpha) # Levy flight noise
        rb = np.random.randn(NP, dim) # Brownian noise
        # cf_new: contraction factor
        cf_new = common.cf_new(iter_idx, Max_t)
        learning_rate = 1 - alpha0 * iter_idx / Max_t # QL learning rate

        # epsilon-greedy action
        if np.random.uniform(0, 1) < eps:
            action = np.random.randint(0, n_action)
        else:
            action = np.argmax(q_table[curr_state, :])

        if action == 0:
            # diff_mut: DE/rand/2 step
            for i in range(NP):
                r_num = random.sample(range(NP), 5)
                for j in range(dim):
                    r_val = np.random.uniform(0, 1)
                    stepsize[i, j] = rb[i, j] * (
                        prey[r_num[0], j]
                        + np.random.uniform(0, 1) * (prey[r_num[1], j] - prey[r_num[2], j])
                        + np.random.uniform(0, 1) * (prey[r_num[3], j] - prey[r_num[4], j])
                    )
                    prey[i, j] = prey[i, j] + P * r_val * stepsize[i, j]

        elif action == 1:
            # dyn_alt
            half = NP // 2
            for i in range(NP):
                # first half of the population
                if i < half:
                    # Levy flight
                    for j in range(dim):
                        r_val = np.random.uniform(0, 1)
                        stepsize[i, j] = rl[i, j] * (elite[i, j] - rl[i, j] * prey[i, j])
                        if np.random.random() < 0.5:
                            # sin
                            prey[i, j] = prey[i, j] + P * r_val * np.sin(
                                2 * np.pi * np.random.uniform(0, 1)
                            ) * stepsize[i, j]
                        else:
                            # cos
                            prey[i, j] = prey[i, j] + P * r_val * np.cos(
                                2 * np.pi * np.random.uniform(0, 1)
                            ) * stepsize[i, j]
                else:
                    # second half of the population
                    for j in range(dim):
                        # Brownian motion
                        stepsize[i, j] = rb[i, j] * (rb[i, j] * elite[i, j] - prey[i, j])
                        if np.random.random() < 0.5:
                            # sin
                            prey[i, j] = elite[i, j] + cf_new * np.sin(
                                2 * np.pi * np.random.uniform(0, 1)
                            ) * stepsize[i, j]
                        else:
                            # cos
                            prey[i, j] = elite[i, j] + cf_new * np.cos(
                                2 * np.pi * np.random.uniform(0, 1)
                            ) * stepsize[i, j]

        else:
            # cobl: candidate vs centroid opposition
            centroid_pop = np.mean(prey, axis=0)
            pos = np.zeros(dim)
            pos_opposite = np.zeros(dim)
            for i in range(NP):
                for j in range(dim):
                    stepsize[i, j] = rl[i, j] * (rl[i, j] * elite[i, j] - prey[i, j])
                    pos[j] = elite[i, j] + P * cf_new * stepsize[i, j]
                    pos_opposite[j] = 2 * centroid_pop[j] - pos[j]
                pos_clip = np.clip(pos, lb, ub)
                pos_opposite_clip = np.clip(pos_opposite, lb, ub)
                if fobj(pos_opposite_clip) < fobj(pos_clip):
                    # update prey if opposite is better
                    prey[i, :] = pos_opposite_clip
                else:
                    # update prey if current is better
                    prey[i, :] = pos_clip

        # re-evaluate after action update
        for i in range(NP):
            prey[i] = np.clip(prey[i], lb, ub)
            all_fitness[i] = fobj(prey[i])
            if all_fitness[i] < top_predator_fit:
                top_predator_fit = all_fitness[i]
                top_predator_pos = prey[i].copy()

        # memory again after action
        if iter_idx == 0:
            fitness_old = all_fitness.copy()
            prey_old = prey.copy()
        inx = fitness_old < all_fitness
        indx = np.tile(inx, (dim, 1)).T
        prey = indx * prey_old + ~indx * prey
        all_fitness = inx * fitness_old + ~inx * all_fitness
        fitness_old = all_fitness.copy()
        prey_old = prey.copy()

        # FADs effect
        if np.random.uniform(0, 1) <= FADs:
            u_mask = np.random.rand(NP, dim) < FADs
            prey = prey + cf_new * (
                (xmin + np.random.rand(NP, dim) * (xmax - xmin)) * u_mask
            )
        else:
            r_val = np.random.uniform(0, 1)
            prey = prey + (FADs * (1 - r_val) + r_val) * (
                prey[np.random.permutation(NP), :]
                - prey[np.random.permutation(NP), :]
            )

        # re-evaluate after FADs
        for i in range(NP):
            prey[i] = np.clip(prey[i], lb, ub)
            all_fitness[i] = fobj(prey[i])
            if all_fitness[i] < top_predator_fit:
                top_predator_fit = all_fitness[i]
                top_predator_pos = prey[i].copy()

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

    return top_predator_fit, convergence_curve, all_reward, q_table
