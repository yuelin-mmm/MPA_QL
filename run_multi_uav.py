"""
Multi-UAV path planning with MPA-QL.

Example:
  python run_multi_uav.py
  python run_multi_uav.py --runtime 1
"""

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mpa_ql"))
sys.path.insert(0, str(ROOT / "env" / "multi_uav"))

import CostFunction  # noqa: E402
import multi_MPA_QL  # noqa: E402
import scenario  # noqa: E402
from load_config import load_config, algo_params  # noqa: E402

OUTPUT_DIR = ROOT / "output"

if __name__ == "__main__":
    # parse command line arguments
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()

    cfg = load_config(pre_args.config) if pre_args.config else load_config()
    mu = cfg["multi_uav"]
    search = cfg["search"]
    params = algo_params(cfg)

    scenario.load()
    parser = argparse.ArgumentParser(description="Multi-UAV MPA-QL", parents=[pre])
    parser.add_argument("--scenario", default=mu["scenario"], choices=list(scenario.ids))
    parser.add_argument("--runtime", type=int, default=mu["runtime"])
    parser.add_argument("--NP", type=int, default=search["NP"])
    parser.add_argument("--Max_t", type=int, default=search["Max_t"])
    args = parser.parse_args()

    # get scenario
    scene = scenario.get_runtime(args.scenario)
    uav_num = scene["uav_num"]

    # output paths
    tag = f"{args.scenario}_{uav_num}"
    txt_path = OUTPUT_DIR / f"0_MPAQL{tag}.txt"
    metrics_path = OUTPUT_DIR / f"metrics_MPAQL{tag}.jsonl"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if txt_path.exists():
        txt_path.unlink()
    if metrics_path.exists():
        metrics_path.unlink()

    # run multi-UAV MPA-QL
    for i in range(args.runtime):
        print(f"scenario={args.scenario} runtime={i}")
        t0 = time.time()
        top_fit, curve, top_pos, _, _, top_len = multi_MPA_QL.multi_MPA_QL(
            args.NP,
            uav_num,
            mu["N_wp"],
            mu["dim"],
            scene["p_start"],
            scene["p_target"],
            scene["lb_array"],
            scene["ub_array"],
            args.Max_t,
            scene["height_max"],
            scene["height_min"],
            scene["scenario_num"],
            mu["Q_coll"],
            mu["phi_max"],
            mu["psi_max"],
            mu["mu1"],
            mu["mu2"],
            mu["mu3"],
            mu["mu4"],
            mu["safe_dis"],
            **params,
        )
        exe_time = time.time() - t0
        log_curve = [float(np.log(x)) for x in curve]

        with txt_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{exe_time}\n")
            fh.write(f"{top_fit}\n")
            fh.write(f"{top_len}\n")
            fh.write(f"{curve}\n")
            fh.write(f"{log_curve}\n")
            fh.write(f"{top_pos}\n")

        metrics = CostFunction.evaluate_solution_metrics(
            top_pos,
            scene["p_start"],
            scene["p_target"],
            scene["height_max"],
            scene["height_min"],
            scene["scenario_num"],
            mu["Q_coll"],
            mu["phi_max"],
            mu["psi_max"],
            mu["mu1"],
            mu["mu2"],
            mu["mu3"],
            mu["mu4"],
            mu["safe_dis"],
        )
        metrics["exe_time"] = exe_time
        metrics["algorithm"] = "MPAQL"
        metrics["scenario"] = args.scenario
        metrics["scenario_num"] = scene["scenario_num"]
        metrics["uav_num"] = uav_num
        metrics["runtime_idx"] = i
        metrics.update({"version": "MPA-QL", **params})

        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(metrics, ensure_ascii=False) + "\n")

        print(
            f"  fit={float(top_fit):.4f} collision_free={metrics['collision_free']} "
            f"time={exe_time:.1f}s -> {txt_path.name}"
        )
