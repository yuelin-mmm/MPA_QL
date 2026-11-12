"""
Multi-UAV path cost: length, altitude, collision, smoothness.
"""

import numpy as np
import scenario

# cache for collision masks
_COLLISION_MASK_CACHE = {}

def _collision_masks(uav_num):
    # skip self-pairs; count each UAV pair once
    if uav_num not in _COLLISION_MASK_CACHE:
        not_self = ~np.eye(uav_num, dtype=bool)
        upper_triangle = np.triu(np.ones((uav_num, uav_num), dtype=bool), k=1)
        _COLLISION_MASK_CACHE[uav_num] = (
            not_self[np.newaxis, :, :, np.newaxis],
            upper_triangle[np.newaxis, :, :, np.newaxis],
        )
    return _COLLISION_MASK_CACHE[uav_num]

def _nfz_violations(middle_points, nfz_list):
    # how many middle waypoints fall in no-fly zones
    if not nfz_list:
        return np.zeros(middle_points.shape[0], dtype=np.intp)

    x = middle_points[:, :, :, 0]
    y = middle_points[:, :, :, 1]
    violations = np.zeros(middle_points.shape[0], dtype=np.intp)
    for box in nfz_list:
        cx = box["cx"]
        cy = box["cy"]
        hx = box["half_x"]
        hy = box["half_y"]
        inside = (
            (x >= cx - hx)
            & (x <= cx + hx)
            & (y >= cy - hy)
            & (y <= cy + hy)
        )
        violations += np.sum(inside, axis=(1, 2))
    return violations

def _collision_components(agents, scenario_num, safe_dis):
    # collision checks
    z_grid = scenario.get_terrain_z(scenario_num)
    nfz_list = scenario.get_nfz(scenario_num)
    middle_points = agents[:, :, 1:-1, :]
    uav_num = agents.shape[1]

    # UAV-UAV collisions
    middle_yz = middle_points[:, :, :, 1:3]
    yz_expanded1 = middle_yz[:, :, np.newaxis, :, :]
    yz_expanded2 = middle_yz[:, np.newaxis, :, :, :]
    distances = np.sqrt(np.sum((yz_expanded1 - yz_expanded2) ** 2, axis=-1))
    not_self, upper_triangle = _collision_masks(uav_num)
    collision_mask = (distances < safe_dis) & not_self & upper_triangle
    inter_uav_collisions = np.sum(collision_mask, axis=(1, 2, 3))

    # terrain collision
    middle_xy = middle_points[:, :, :, :2].astype(np.intp)
    x_coords = np.clip(middle_xy[:, :, :, 0] - 1, 0, z_grid.shape[0] - 1)
    y_coords = np.clip(middle_xy[:, :, :, 1] - 1, 0, z_grid.shape[1] - 1)
    obs_height = z_grid[x_coords, y_coords]
    uav_height = middle_points[:, :, :, 2]
    terrain_collisions = np.sum(uav_height < obs_height, axis=(1, 2))
    
    # NFZ collision
    nfz_collisions = _nfz_violations(middle_points, nfz_list)
    
    # minimum clearance to terrain (min height above terrain)
    min_clearance = np.min(uav_height - obs_height, axis=(1, 2))

    return inter_uav_collisions, terrain_collisions, nfz_collisions, min_clearance

def _heading_pitch_deg(agents):
    # heading and pitch angles at each waypoint (degrees)
    prev_points = agents[:, :, :-2, :]
    curr_points = agents[:, :, 1:-1, :]
    next_points = agents[:, :, 2:, :]

    vec1_hori = curr_points[:, :, :, :2] - prev_points[:, :, :, :2]
    vec2_hori = next_points[:, :, :, :2] - curr_points[:, :, :, :2]
    cross_mag = np.abs(
        vec1_hori[:, :, :, 0] * vec2_hori[:, :, :, 1]
        - vec1_hori[:, :, :, 1] * vec2_hori[:, :, :, 0]
    )
    dot_prod = np.sum(vec1_hori * vec2_hori, axis=3)
    headings = np.arctan2(cross_mag, dot_prod)

    dz = next_points[:, :, :, 2] - curr_points[:, :, :, 2]
    dist_hori = np.sqrt(np.sum(vec2_hori ** 2, axis=3))
    dist_hori = np.maximum(dist_hori, 1e-10)
    pitches = np.arctan2(dz, dist_hori)

    return np.rad2deg(np.abs(headings)), np.rad2deg(np.abs(pitches))

def select_better_prey(
    prey_a, prey_b, all_length, height_max, height_min, scenario_num, Q_coll,
    phi_max, psi_max, mu1, mu2, mu3, mu4, safe_dis):
    # COBL: keep candidate or opposite, whichever has lower total cost
    n = prey_a.shape[0]
    stacked = np.concatenate([prey_a, prey_b], axis=0)
    fitness_stack, _ = cost_sum(
        stacked, all_length, height_max, height_min, scenario_num, Q_coll,
        phi_max, psi_max, mu1, mu2, mu3, mu4, safe_dis)
    fitness_a = fitness_stack[:n]
    fitness_b = fitness_stack[n:]
    mask = (fitness_b < fitness_a).reshape(-1, 1, 1, 1)
    return np.where(mask, prey_b, prey_a)

def cost_length(agents, all_length):
    # path-length term (vs start-goal distance) and total length (for logging)
    diff = np.diff(agents, axis=2)
    segment_dists = np.sqrt(np.sum(diff ** 2, axis=3))
    normalized_lengths = np.sum(segment_dists, axis=2) / all_length
    return np.sum(normalized_lengths, axis=1), np.sum(np.sum(segment_dists, axis=2), axis=1)

def cost_altitude(agents, height_max, height_min):
    # altitude bound violation
    excess_max = np.maximum(agents[:, :, 1:-1, 2] - height_max, 0)
    excess_min = np.maximum(height_min - agents[:, :, 1:-1, 2], 0)
    total_excess = np.maximum(excess_max, excess_min)
    return np.sum(total_excess, axis=(1, 2)) / (height_max + height_min)

def cost_collision(agents, scenario_num, Q_coll, safe_dis):
    # collision penalty (UAV-UAV / terrain / NFZ)
    inter_uav, terrain, nfz, _ = _collision_components(agents, scenario_num, safe_dis)
    return Q_coll * (inter_uav + terrain + nfz)

def cost_smooth(agents, phi_max, psi_max):
    # heading / pitch excess beyond limits
    headings_deg, pitches_deg = _heading_pitch_deg(agents)
    heading_cost = np.where(headings_deg > phi_max, headings_deg - phi_max, 0)
    pitch_cost = np.where(pitches_deg > psi_max, pitches_deg - psi_max, 0)
    return (
        np.sum(heading_cost, axis=(1, 2)) / (agents.shape[2] - 2)
        + np.sum(pitch_cost, axis=(1, 2)) / (agents.shape[2] - 1))

def heading_pitch_statistics(agents, phi_max, psi_max):
    # heading / pitch stats for logging
    headings_deg, pitches_deg = _heading_pitch_deg(agents)
    return {
        "heading_avg": np.mean(headings_deg, axis=(1, 2)),
        "heading_std": np.std(headings_deg, axis=(1, 2)),
        "heading_violation_count": np.sum(headings_deg > phi_max, axis=(1, 2)),
        "pitch_avg": np.mean(pitches_deg, axis=(1, 2)),
        "pitch_std": np.std(pitches_deg, axis=(1, 2)),
        "pitch_violation_count": np.sum(pitches_deg > psi_max, axis=(1, 2)),
    }

def cost_sum(
    agents, all_length, height_max, height_min, scenario_num, Q_coll,
    phi_max, psi_max, mu1, mu2, mu3, mu4, safe_dis):
    # J = mu1*J_len + mu2*J_alt + mu3*J_coll + mu4*J_smooth
    J_len, length = cost_length(agents, all_length)
    J_alt = cost_altitude(agents, height_max, height_min)
    J_coll = cost_collision(agents, scenario_num, Q_coll, safe_dis)
    J_smooth = cost_smooth(agents, phi_max, psi_max)
    return (
        mu1 * J_len + mu2 * J_alt + mu3 * J_coll + mu4 * J_smooth,
        length)

def evaluate_solution_metrics(
    best_position, p_start, p_target, height_max, height_min, scenario_num, Q_coll,
    phi_max, psi_max, mu1, mu2, mu3, mu4, safe_dis):
    # summary metrics of the best path (for output logs)
    agents = best_position[np.newaxis, :, :, :]
    all_length = np.linalg.norm(p_target - p_start)
    fitness, lengths = cost_sum(
        agents, all_length, height_max, height_min, scenario_num, Q_coll,
        phi_max, psi_max, mu1, mu2, mu3, mu4, safe_dis)
    inter_collisions, terrain_collisions, nfz_collisions, min_clearance = (
        _collision_components(agents, scenario_num, safe_dis))
    smooth_stats = heading_pitch_statistics(agents, phi_max, psi_max)

    middle_z = agents[:, :, 1:-1, 2]
    altitude_violation_count = np.sum(
        (middle_z > height_max) | (middle_z < height_min), axis=(1, 2)
    )

    collision_free = (
        inter_collisions[0] == 0
        and terrain_collisions[0] == 0
        and nfz_collisions[0] == 0
    )

    return {
        "fitness": float(fitness[0]),
        "path_length": float(lengths[0]),
        "inter_uav_collision_count": int(inter_collisions[0]),
        "obstacle_collision_count": int(terrain_collisions[0]),
        "nfz_collision_count": int(nfz_collisions[0]),
        "collision_free": bool(collision_free),
        "min_clearance_to_obstacle": float(min_clearance[0]),
        "heading_avg": float(smooth_stats["heading_avg"][0]),
        "heading_std": float(smooth_stats["heading_std"][0]),
        "heading_violation_count": int(smooth_stats["heading_violation_count"][0]),
        "pitch_avg": float(smooth_stats["pitch_avg"][0]),
        "pitch_std": float(smooth_stats["pitch_std"][0]),
        "pitch_violation_count": int(smooth_stats["pitch_violation_count"][0]),
        "altitude_avg": float(np.mean(middle_z)),
        "altitude_std": float(np.std(middle_z)),
        "altitude_violation_count": int(altitude_violation_count[0]),
    }
