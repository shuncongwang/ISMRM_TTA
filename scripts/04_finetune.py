#!/usr/bin/env python3
"""
Step 4 - supervised fine-tuning upper bound: start from the source model and train on the
n_finetune labelled cases (role target_finetune) of ONE target domain.  Evaluate it afterwards with
05_run_tta.py --methods source --ckpt <this model> --label finetune10.

    python scripts/04_finetune.py --config config.yaml --split exp1_cohort_src-UCSF-PDGM --domain EGD
    python scripts/04_finetune.py --config config.yaml --split exp1_cohort_src-UCSF-PDGM --domain all

Model goes to <work_dir>/models/<split>/finetune_<domain>/checkpoint_final.pt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tta.common import LOG, load_config, read_split, seed_everything, setup_logging
from tta.train_loop import train


def safe(s):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)


def main():
    def extra(ap):
        ap.add_argument("--split", required=True)
        ap.add_argument("--domain", required=True, help="target domain name, or 'all'")
        ap.add_argument("--source_tag", default="")
    cfg, a = load_config(extra_args=extra)
    src_dir = Path(cfg["work_dir"]) / "models" / (a.split + (f"_{a.source_tag}" if a.source_tag else ""))
    setup_logging(src_dir / "finetune.log")
    seed_everything(cfg["seed"])
    sp = read_split(Path(cfg["work_dir"]) / "splits" / f"{a.split}.csv")
    ft = sp[sp.role == "target_finetune"]
    domains = sorted(ft.domain.unique()) if a.domain == "all" else [a.domain]
    for dom in domains:
        ids = ft[ft.domain == dom].case_id.tolist()
        if not ids:
            LOG.info(f"{dom}: no target_finetune cases (domain too small) - skipped")
            continue
        out = src_dir / f"finetune_{safe(dom)}"
        LOG.info(f"fine-tuning on {dom}: {len(ids)} cases, {cfg['finetune_iters']} iterations")
        # no validation set (only 10 labelled cases): fixed iteration budget, final checkpoint used
        cfg_ft = dict(cfg, val_every=10 ** 9)
        epochs = max(1, cfg["finetune_iters"] // 50)
        train(cfg_ft, ids, [], out, epochs, 50, cfg["finetune_lr"], init_ckpt=src_dir / "checkpoint_best.pt",
              tag="finetune")


if __name__ == "__main__":
    main()
