# MPA-QL

*A Q-Learning-Enhanced Marine Predator Algorithm for Multi-UAV Path Planning* (PRICAI 2026).

🔗 The paper link will be available soon.

## Layout

```text
MPA_QL/
  config.yaml              # experiment / algorithm defaults
  load_config.py           # yaml -> algorithm kwargs
  mpa_ql/                  # MPA_QL(for CEC), multi_MPA_QL(for multi-UAV)
  env/
    multi_uav/             # CostFunction, scenario.py, scenario_example.json
    cec2022/               # CEC2022.py, input_data/, cec2022_bridge.py
  run_multi_uav.py
  run_cec.py
```

Run logs go to `output/` (multi-UAV) and `output_cec/` (CEC).

## Install

```bash
pip install -r requirements.txt
```

## Configs

Default configurations are in [`config.yaml`](config.yaml). CLI flags override them when provided.

## How to Run

### Multi-UAV Env

```bash
python run_multi_uav.py
python run_multi_uav.py --runtime 1          # smoke
```

Outputs: `output/0_*.txt`, `output/metrics_*.jsonl`.

#### Scene JSON

Edit [`env/multi_uav/scenario_example.json`](env/multi_uav/scenario_example.json) to adjust scene parameters.

| Parameter | Meaning |
|-------|---------|
| `id` | Scene name (must match `config.yaml` / `--scenario`) |
| `map_size` | Square map side length (m), e.g. `300` means 300×300 |
| `z_max` | Vertical search upper bound (m) |
| `height_min` / `height_max` | Flight altitude band (m) |
| `uav_num` | Number of UAVs |
| `p_start` / `p_target` | Start / goal `[x, y, z]` |
| `n_peaks` | Peak count (must match `peaks` list lengths) |
| `peaks` | `h_list`, `Xc_list`, `Yc_list`, `Xs_list`, `Ys_list` |
| `nfz_count` | NFZ count (must match `nfz` length) |
| `nfz` | Rectangles: `cx`, `cy`, `half_x`, `half_y` |

### CEC2022 Benchmark

```bash
python run_cec.py
python run_cec.py --dim 10 --start 1 --end 3 --runtime 3   # smoke
```

Outputs: `output_cec/MPAQL_dim{10|20}.txt`.

#### Source

`env/cec2022/CEC2022.py` and `input_data/` are the official Python CEC 2022 bound-constrained suite (Abhishek Kumar et al.), from [P-N-Suganthan/2022-SO-BO](https://github.com/P-N-Suganthan/2022-SO-BO) (`Python-CEC2022.zip`).
