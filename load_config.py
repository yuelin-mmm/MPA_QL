"""
Load MPA_QL/config.yaml.
"""

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.yaml"

def load_config(path=DEFAULT_CONFIG):
    # load yaml config
    cfg_path = Path(path)
    with cfg_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)

def algo_params(cfg):
    # algorithm parameters from config
    q = cfg["mpa_ql"]
    m = cfg["mpa"]
    ql = cfg["q_learning"]
    return {
        "rho": q["rho"],
        "H": q["H"],
        "tau": q["tau"],
        "n_state": q["n_state"],
        "n_action": q["n_action"],
        "P": m["P"],
        "FADs": m["FADs"],
        "levy_scale": m["levy_scale"],
        "levy_alpha": m["levy_alpha"],
        "eps": ql["eps"],
        "gamma": ql["gamma"],
        "alpha0": ql["alpha0"],
    }
