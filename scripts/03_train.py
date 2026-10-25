#!/usr/bin/env python3
"""
Step 3 - train the source model of one experiment (cases with role source_train; source_val for
model selection).

    python scripts/03_train.py --config config.yaml --split exp1_cohort_src-UCSF-PDGM
    python scripts/03_train.py --config config.yaml --split exp1_cohort_src-UCSF-PDGM --seed 1 --tag seed1
    python scripts/03_train.py ... --resume          # continue after a time-limit kill

Model goes to <work_dir>/models/<split>[_<tag>]/checkpoint_best.pt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tta.common import LOG, load_config, read_split, seed_everything, setup_logging
from tta.train_loop import train


def main():
    def extra(ap):
        ap.add_argument("--split", required=True, help="split name (file in <work_dir>/splits without .csv)")
        ap.add_argument("--tag", default="")
        ap.add_argument("--resume", action="store_true")
    cfg, a = load_config(extra_args=extra)
    name = a.split + (f"_{a.tag}" if a.tag else "")
    out = Path(cfg["work_dir"]) / "models" / name
    setup_logging(out / "train.log")
    seed_everything(cfg["seed"])
    sp = read_split(Path(cfg["work_dir"]) / "splits" / f"{a.split}.csv")
    tr = sp[sp.role == "source_train"].case_id.tolist()
    va = sp[sp.role == "source_val"].case_id.tolist()
    LOG.info(f"{name}: {len(tr)} train / {len(va)} val cases, preset {cfg['preset']}, patch {cfg['patch_size']}")
    best = train(cfg, tr, va, out, cfg["epochs"], cfg["iters_per_epoch"], cfg["lr"], resume=a.resume)
    LOG.info(f"done -> {best}")


if __name__ == "__main__":
    main()
