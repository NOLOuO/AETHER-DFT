from __future__ import annotations

import copy
import logging
import tomllib
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_CONFIG_PATH = PROJECT_ROOT / "workflow_config.toml"
LOGGER = logging.getLogger(__name__)

DEFAULT_WORKFLOW_CONFIG: dict[str, Any] = {
    "preopt": {
        "mace": {
            "model": "medium-mpa-0",
            "dtype": "float32",
            "fmax": 0.05,
            "max_steps": 500,
            "optimizer": "lbfgs",
        },
        "cluster": {
            "queue": "c-node",
            "cores": 32,
            "vasp_version": "std",
            "nodes": 1,
        },
    },
    "ts": {
        "cluster": {
            "queue": "c-node",
            "cores": 32,
            "vasp_version": "std",
            "nodes": 1,
        },
        "neb": {
            "n_images": 8,
            "k_spring": 0.1,
            "use_climb": True,
            "anchor_layers": 2,
            "z_tol": 0.15,
            "stage1_only": False,
            "force_same_substrate_from_fs": False,
            "substrate_sync_layers": 4,
            "use_remove_xy_drift": True,
            "drift_ref_layers": 4,
            "fmax_fast": 0.10,
            "steps_fast": 200,
            "opt_fast": "FIRE",
            "fmax_refine": 0.10,
            "steps_refine": 500,
            "opt_refine": "LBFGS",
            "model": "medium-mpa-0",
        },
        "freq": {
            "model": "medium-mpa-0",
            "dtype": "float64",
            "delta": 0.01,
            "imag_threshold_cm1": 50.0,
            "soft_imag_max_cm1": 120.0,
            "max_significant_imag_modes_for_ts": 1,
            "cache_dir": "vib_cache",
        },
        "modecar": {
            "freq_result_file": "freq_result.json",
            "vib_name": "vib_cache",
            "output_file": "MODECAR",
            "imag_threshold_eV": 1e-4,
        },
    },
}


def load_workflow_config() -> dict[str, Any]:
    payload = copy.deepcopy(DEFAULT_WORKFLOW_CONFIG)
    if not WORKFLOW_CONFIG_PATH.exists():
        return payload
    try:
        raw = tomllib.loads(WORKFLOW_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("读取 workflow_config.toml 失败，将回退到默认配置: %s", exc)
        return payload
    _deep_merge(payload, raw)
    return payload


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = value
