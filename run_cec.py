"""
CEC2022 numerical optimization with MPA-QL.

Example:
  python run_cec.py
  python run_cec.py --dim 10 --runtime 3 --start 1 --end 3
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mpa_ql"))
sys.path.insert(0, str(ROOT / "env" / "cec2022"))

import cec2022_bridge  # noqa: E402
from MPA_QL import MPA_QL  # noqa: E402
from load_config import load_config, algo_params  # noqa: E402

OUTPUT_CEC = ROOT / "output_cec"

if __name__ == "__main__":
    # parse command line arguments
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()

    cfg = load_config(pre_args.config) if pre_args.config else load_config()
    cec = cfg["cec"]
    search = cfg["search"]
    params = algo_params(cfg)
    n_func = cec2022_bridge.num_functions()
    end_default = cec["end"] if cec["end"] is not None else n_func + 1

    parser = argparse.ArgumentParser(description="MPA-QL CEC2022", parents=[pre])
    parser.add_argument("--dim", type=int, default=cec["dim"], choices=(10, 20))
    parser.add_argument("--runtime", type=int, default=cec["runtime"])
    parser.add_argument("--start", type=int, default=cec["start"])
    parser.add_argument("--end", type=int, default=end_default)
    parser.add_argument("--NP", type=int, default=search["NP"])
    parser.add_argument("--Max_t", type=int, default=search["Max_t"])
    args = parser.parse_args()

    # validate function range: 1 <= start < end <= n_func + 1
    if not (1 <= args.start <= n_func and 2 <= args.end <= n_func + 1 and args.start < args.end):
        raise ValueError(f"start/end must satisfy: 1 <= start < end <= {n_func + 1}")

    OUTPUT_CEC.mkdir(parents=True, exist_ok=True)
    lb, ub = cec2022_bridge.bounds()
    file_path = OUTPUT_CEC / f"MPAQL_dim{args.dim}.txt"
    print(f"output -> {file_path}")

    # run each function in turn
    for func_idx in range(args.start, args.end):
        print(f"dim={args.dim} function={func_idx}")
        with file_path.open("a", encoding="utf-8") as fh:
            fh.write(f"Function{func_idx}\n")

        fobj = cec2022_bridge.make_objective(func_idx, args.dim)
        for run_idx in range(args.runtime):
            print(f"  runtime={run_idx}")
            t0 = time.time()
            top_fit, curve, _, _ = MPA_QL(
                fobj, args.NP, args.Max_t, lb, ub, args.dim, **params,
            )
            exe_time = time.time() - t0
            log_curve = [float(np.log(max(x, 1e-30))) for x in curve]
            with file_path.open("a", encoding="utf-8") as fh:
                fh.write(f"{exe_time}\n")
                fh.write(f"{top_fit}\n")
                fh.write(f"{curve}\n")
                fh.write(f"{log_curve}\n")
