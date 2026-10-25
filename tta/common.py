"""Shared helpers: configuration, logging, case tables."""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

LOG = logging.getLogger("glioma_tta")


def setup_logging(log_file: Path | None = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=handlers, force=True)


def _auto(s):
    for t in (int, float):
        try:
            return t(s)
        except ValueError:
            pass
    return s


def load_config(argv=None, extra_args=None) -> tuple[dict, argparse.Namespace]:
    """Read --config YAML, then let any top-level key be overridden as --key value."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yaml"))
    known, rest = pre.parse_known_args(argv)
    cfg = yaml.safe_load(Path(known.config).read_text())

    ap = argparse.ArgumentParser(parents=[pre])
    if extra_args:
        extra_args(ap)
    for k, v in cfg.items():
        if isinstance(v, dict):
            continue
        if isinstance(v, list):
            ap.add_argument(f"--{k}", nargs="+", default=v,
                            type=type(v[0]) if v and not isinstance(v[0], str) else str)
        elif isinstance(v, bool):
            ap.add_argument(f"--{k}", default=v, type=lambda s: s.lower() in {"1", "true", "yes"})
        elif v is None:  # e.g. target_spacing, wt_labels, tta_patch_size: accept a list on the command line
            ap.add_argument(f"--{k}", nargs="+", default=None, type=_auto)
        else:
            ap.add_argument(f"--{k}", default=v, type=type(v))
    args = ap.parse_args(rest)
    for k in cfg:
        if not isinstance(cfg[k], dict) and hasattr(args, k):
            cfg[k] = getattr(args, k)
    for k, v in cfg.items():  # "--x null" on the command line
        if isinstance(v, list) and len(v) == 1 and str(v[0]).lower() in {"null", "none"}:
            cfg[k] = None
    base = Path(known.config).resolve().parent  # relative paths in the config are relative to its folder
    for k in ("data_root", "merged_xlsx", "work_dir"):
        if cfg.get(k):
            q = Path(str(cfg[k])).expanduser()
            cfg[k] = str(q if q.is_absolute() else (base / q).resolve())
    return cfg, args


def amp_dtype(enabled: bool, device: str):
    """Mixed-precision type for this GPU:
    bf16 on Ampere or newer (A40, A100; compute capability >= 8),
    fp16 + loss scaling on Volta/Turing (V100, RTX 6000/8000; 7.x, tensor cores),
    None = full fp32 on Pascal (P6000, P100; 6.x: fp16 maths is not faster there) and on CPU."""
    if not enabled or not str(device).startswith("cuda"):
        return None
    import torch
    major = torch.cuda.get_device_capability()[0]
    if major >= 8:
        return torch.bfloat16
    if major == 7:
        return torch.float16
    return None


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def read_table(path, sheet=None) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls", ".xlsm"}:
        xl = pd.ExcelFile(path)
        df = xl.parse(sheet if sheet in xl.sheet_names else xl.sheet_names[0], dtype=object)
    else:
        df = pd.read_csv(path, dtype=object, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def truthy(v) -> bool:
    return str(v).strip().lower() in {"1", "1.0", "true", "yes", "y", "t"}


def cache_dir(cfg) -> Path:
    return Path(cfg["work_dir"]) / "cache"


def case_files(cfg, case_id: str) -> dict:
    d = cache_dir(cfg)
    return {"img": d / f"{case_id}_img.npy", "seg": d / f"{case_id}_seg.npy",
            "brain": d / f"{case_id}_brain.npy", "meta": d / f"{case_id}_meta.json"}


def load_case(cfg, case_id: str, mmap=True) -> dict:
    f = case_files(cfg, case_id)
    mode = "r" if mmap else None
    return {"img": np.load(f["img"], mmap_mode=mode), "seg": np.load(f["seg"], mmap_mode=mode),
            "brain": np.load(f["brain"], mmap_mode=mode), "meta": json.loads(f["meta"].read_text())}


def read_split(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    return df.fillna("")
