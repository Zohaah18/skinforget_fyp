"""Fine-tune-only control: skip noise + impair, run only the repair step on the retain set."""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from config_loader import load_cfg
from dataset import CLASS_TO_IDX, HAMDataset, balanced_sampler, get_transforms
from unlearn import FORGET_SETS, make_opt, param_drift, summarize, sync, train_epochs
from utils import atomic_save, build_model, evaluate, load_weights, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--forget", nargs="+", default=["C1", "C2", "C4", "Cm"], choices=list(FORGET_SETS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = bool(cfg.get("use_amp", True)) and device == "cuda"
    d, sz = cfg["splits_dir"], cfg["image_size"]
    train_eval = HAMDataset(f"{d}/train.csv", cfg["data_root"], get_transforms(False, sz))
    test_ds = HAMDataset(f"{d}/test.csv", cfg["data_root"], get_transforms(False, sz))
    test_dl = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                         num_workers=cfg["num_workers"])
    labels = np.array(train_eval.labels)

    for tag in args.forget:
        run_id = f"finetune_{tag}_seed{args.seed}"
        out_dir = Path(cfg.get("out_dir", "outputs")) / "references" / run_id
        if (out_dir / "metrics.json").exists() and not args.force:
            print(f"[skip] {run_id} already done")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        set_seed(args.seed)

        forget_idx = [CLASS_TO_IDX[c] for c in FORGET_SETS[tag]]
        r_ids = np.where(~np.isin(labels, forget_idx))[0].tolist()
        model = load_weights(build_model(pretrained=False), args.baseline, device).to(device)
        orig = copy.deepcopy(model).to(device).eval()
        y, p, _ = evaluate(model, test_dl, device, amp=amp)
        base = summarize(y, p, forget_idx)

        r_ds = Subset(train_eval, r_ids)
        if cfg["repair_balanced"]:
            dl = DataLoader(r_ds, batch_size=cfg["batch_size"],
                            sampler=balanced_sampler([labels[i] for i in r_ids]),
                            num_workers=cfg["num_workers"])
        else:
            dl = DataLoader(r_ds, batch_size=cfg["batch_size"], shuffle=True,
                            num_workers=cfg["num_workers"])
        sync(device); t0 = time.time()
        opt = make_opt(cfg["repair_optim"], model.parameters(), cfg["repair_lr"])
        train_epochs(model, dl, opt, device, cfg["repair_epochs"], amp)
        sync(device); t_ft = time.time() - t0
        y, p, _ = evaluate(model, test_dl, device, amp=amp)
        rec = summarize(y, p, forget_idx)

        res = {"run_id": run_id, "tag": tag, "seed": args.seed, "base": base, "recovered": rec,
               "time_s": t_ft, "MPD": param_drift(model, orig)}
        json.dump(res, open(out_dir / "metrics.json", "w"), indent=2)
        atomic_save({"model": model.state_dict()}, out_dir / "final.pt")
        row = dict(run_id=run_id, tag=tag, time_s=t_ft, MPD=res["MPD"],
                   Ai_full=base["acc_full"], Ar_full=rec["acc_full"],
                   Ai_retain=base["acc_retain"], Ar_retain=rec["acc_retain"],
                   forget_acc_before=base["acc_forget"], forget_acc_after=rec["acc_forget"])
        csv = Path(cfg.get("out_dir", "outputs")) / "results_references.csv"
        pd.DataFrame([row]).to_csv(csv, mode="a", header=not csv.exists(), index=False)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, indent=1))


if __name__ == "__main__":
    main()