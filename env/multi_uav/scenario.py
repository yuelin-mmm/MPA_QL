"""
Multi-UAV scene: terrain + load scenario_example.json.
"""

import json
from pathlib import Path
import numpy as np

_json_path = Path(__file__).resolve().parent / "scenario_example.json"
_cfg_by_id = {}
_terrain_by_num = {}
_nfz_by_num = {}
_loaded = False
ids = ()  # scenario IDs from JSON

def _terrain1(X, Y, a1, a2, a3, a4, a5, a6, a7):
    # base terrain Z1 (paper a1–a7 layout)
    r = np.sqrt(X ** 2 + Y ** 2)
    return (
        np.sin(Y + a1) + a2 * np.sin(X)
        + a3 * np.cos(a4 * r)
        + a5 * np.cos(Y)
        + a6 * np.sin(a6 * r)
        + a7 * np.cos(Y)
    )

def _terrain2(X, Y, N, h_list, Xc_list, Yc_list, Xs_list, Ys_list):
    # Gaussian mountain peaks
    Z = np.zeros((len(X), len(X)))
    for i in range(N):
        Z += h_list[i] * np.exp(
            -((X - Xc_list[i]) / Xs_list[i]) ** 2
            - ((Y - Yc_list[i]) / Ys_list[i]) ** 2
        )
    return Z

def _build_terrain(map_size, n_peaks, peaks):
    # build terrain grid: Z = max(Z1, Z2)
    x = np.arange(map_size)
    y = np.arange(map_size)
    x_grid, y_grid = np.meshgrid(x, y)
    # a1=3π, a2=0.1, a3=0.5, a4=0.3, a5=0.9, a6=0.5, a7=0.5
    z1 = _terrain1(
        x_grid, y_grid, 3 * np.pi, 0.1, 0.5, 0.3, 0.9, 0.5, 0.5,
    )
    z2 = _terrain2(
        x_grid, y_grid, n_peaks,
        peaks["h_list"], peaks["Xc_list"], peaks["Yc_list"],
        peaks["Xs_list"], peaks["Ys_list"],
    )
    return np.maximum(z1, z2)

def _ensure_loaded():
    # lazy-load JSON (CEC path does not need scene files)
    global _loaded
    if not _loaded:
        load()

def load(json_path=None):
    # load scene configs / terrain / NFZ from JSON
    global _loaded, ids
    path = Path(json_path) if json_path is not None else _json_path
    with path.open("r", encoding="utf-8") as fh:
        configs = json.load(fh)

    _cfg_by_id.clear()
    _terrain_by_num.clear()
    _nfz_by_num.clear()

    for idx, cfg in enumerate(configs, start=1):
        sid = cfg["id"]
        _cfg_by_id[sid] = cfg
        _terrain_by_num[idx] = _build_terrain(
            cfg["map_size"], cfg["n_peaks"], cfg["peaks"],
        )
        _nfz_by_num[idx] = list(cfg["nfz"])[: cfg["nfz_count"]]

    ids = tuple(cfg["id"] for cfg in configs)
    _loaded = True

def _resolve_id(scenario_id):
    # case-insensitive match against loaded ids
    _ensure_loaded()
    key = scenario_id.casefold()
    for sid in ids:
        if sid.casefold() == key:
            return sid
    raise ValueError(f"Unknown scenario id: {scenario_id}; expected one of {ids}")

def get_terrain_z(scenario_num):
    # terrain height grid for scenario_num
    _ensure_loaded()
    return _terrain_by_num[scenario_num]

def get_nfz(scenario_num):
    # NFZ boxes for scenario_num
    _ensure_loaded()
    return _nfz_by_num[scenario_num]

def get_runtime(scenario_id):
    # optimizer runtime fields from JSON
    sid = _resolve_id(scenario_id)
    cfg = _cfg_by_id[sid]
    map_size = cfg["map_size"]
    z_max = cfg["z_max"]
    return {
        "id": sid,
        "scenario_num": ids.index(sid) + 1,
        "uav_num": cfg["uav_num"],
        "p_start": np.array(cfg["p_start"], dtype=float),
        "p_target": np.array(cfg["p_target"], dtype=float),
        "lb_array": np.array([0.0, 0.0, 0.0]),
        "ub_array": np.array([map_size, map_size, z_max], dtype=float),
        "height_min": cfg["height_min"],
        "height_max": cfg["height_max"],
    }
