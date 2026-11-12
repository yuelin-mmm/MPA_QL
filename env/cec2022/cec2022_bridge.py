"""
Interface to the CEC2022 benchmark suite for scalar objective evaluation.
"""

import os
from pathlib import Path
import numpy as np
from CEC2022 import cec2022_func

CEC_ROOT = Path(__file__).resolve().parent

# CEC2022 suite settings used in this pack
_NUM_FUNCS = 12  # F1-F12
_LB, _UB = -100.0, 100.0  # box bounds for all functions

def num_functions():
    # number of CEC2022 functions (F1-F12)
    return _NUM_FUNCS

def bounds():
    # search bounds (lb, ub) shared by all CEC2022 functions
    return _LB, _UB

def make_objective(func_num, dim):
    # return f(x) for one CEC2022 function (func_num in 1..12, dim in {10,20})
    # CEC2022.py loads input_data/ via relative paths
    os.chdir(str(CEC_ROOT))
    evaluator = cec2022_func(func_num=func_num)

    def fobj(x):
        xv = np.asarray(x, dtype=float).reshape(dim, 1)
        evaluator.values(xv)
        return float(evaluator.ObjFunc[0])

    return fobj
