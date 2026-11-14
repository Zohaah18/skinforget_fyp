"""Ablation: forget-suppression repair ONLY (no noise generation, no impair step).

Answers: does the noise-maximisation step add anything beyond retain-fine-tuning + suppression?
  python src/suppress_only.py --baseline B --forget C1 C2 Cm --seed 0
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from adaptive_unlearn import cycle, objective, repair_epoch
from config_loader import load_cfg
from dataset import CLASS_TO_IDX, CLASSES, HAMDataset, balanced_sampler, get_transforms
from unlearn import FORGET_SETS, make_opt, param_drift, summarize, sync
from utils import atomic_save, build_model, evaluate, load_weights, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--forget", nargs="+", default=["C1", "C2", "Cm"], choices=list(FORGET_SETS))
    ap.add_argument("--suppress", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--note", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = bool(cfg.get("use_amp", True)) and device == "cuda"
    d, sz = cfg["splits_dir"], cfg["image_size"]
    tf = get_transforms(False, sz)
    train_eval = HAMDataset(f"{d}/train.csv", cfg["data_root"], tf)
    kw = dict(batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    val_dl = DataLoader(HAMDataset(f"{d}/val.csv", cfg["data_root"], tf), **kw)
    test_dl = DataLoader(HAMDataset(f"{d}/test.csv", cfg["data_root"], tf), **kw)
    labels = np.array(train_eval.labels)

    for tag in args.forget:
        suffix = (f"_nonoise_sup{args.suppress}" + ("" if args.epochs == 1 else f"_ep{args.epochs}")
                  + args.note)
        run_id = f"resnet50_{tag}_seed{args.seed}{suffix}"
        out_dir = Path(cfg.get("out_dir", "outputs")) / "unlearning" / run_id
        if (out_dir / "metrics.json").exists() and not args.force:
            print(f"[skip] {run_id} already done")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        set_seed(args.seed)
        print(f"\n=== {run_id}: forget {FORGET_SETS[tag]} (suppression-only repair) ===")

        forget_idx = [CLASS_TO_IDX[c] for c in FORGET_SETS[tag]]
        keep_idx = [k for k in range(len(CLASSES)) if k not in forget_idx]
        f_ids = np.where(np.isin(labels, forget_idx))[0].tolist()
        r_ids = np.where(~np.isin(labels, forget_idx))[0].tolist()
        rho = len(f_ids) / len(labels)

        model = load_weights(build_model(pretrained=False), args.baseline, device).to(device)
        orig = copy.deepcopy(model).to(device).eval()
        y, p, _ = evaluate(model, test_dl, device, amp=amp)
        base_t = summarize(y, p, forget_idx)
        y, p, _ = evaluate(model, val_dl, device, amp=amp)
        base_v = summarize(y, p, forget_idx)

        r_ds = Subset(train_eval, r_ids)
        if cfg["repair_balanced"]:
            retain_dl = DataLoader(r_ds, batch_size=cfg["batch_size"],
                                   sampler=balanced_sampler([labels[i] for i in r_ids]),
                                   num_workers=cfg["num_workers"])
        else:
            retain_dl = DataLoader(r_ds, batch_size=cfg["batch_size"], shuffle=True,
                                   num_workers=cfg["num_workers"])
        forget_iter = cycle(DataLoader(Subset(train_eval, f_ids), batch_size=cfg["batch_size"],
                                       shuffle=True, num_workers=cfg["num_workers"]))
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        opt = make_opt(cfg["repair_optim"], model.parameters(), cfg["repair_lr"])
        keep_t = torch.tensor(keep_idx, device=device)

        t_repair = 0.0
        for _ in range(args.epochs):
            sync(device)
            t0 = time.time()
            repair_epoch(model, retain_dl, forget_iter, opt, scaler, device, amp, keep_t, args.suppress)
            sync(device)
            t_repair += time.time() - t0

        y, p, _ = evaluate(model, val_dl, device, amp=amp)
        rec_v = summarize(y, p, forget_idx)
        y, p, _ = evaluate(model, test_dl, device, amp=amp)
        rec_t = summarize(y, p, forget_idx)
        mpd = param_drift(model, orig)
        J, j_f, j_r, j_m = objective(base_v, rec_v, mpd, cfg)

        res = {"run_id": run_id, "tag": tag, "family": "nonoise+sup", "rho": rho,
               "suppress_eta": args.suppress, "J_val": J, "MPD": mpd,
               "times": {"noise": 0.0, "impair": 0.0, "repair": t_repair},
               "val": {"base": base_v, "impair": base_v, "final": rec_v},
               "test": {"base": base_t, "impair": base_t, "final": rec_t}}
        json.dump(res, open(out_dir / "metrics.json", "w"), indent=2)
        atomic_save({"model": model.state_dict()}, out_dir / "final.pt")

        row = dict(run_id=run_id, suffix=suffix, tag=tag, family="nonoise+sup", gamma=0.0, kappa=0.0,
                   rho=rho, alpha=0.0, lam=0.0, beta=0.0, vol=False, eta=args.suppress,
                   n_noisy=0, repair_epochs_used=args.epochs,
                   J_val=J, J_forget=j_f, J_retain_pen=j_r, J_mpd_pen=j_m, MPD=mpd,
                   noise_time_s=0.0, impair_time_s=0.0, repair_time_s=t_repair,
                   total_time_s=t_repair,
                   Af_val_base=base_v["acc_forget"], Af_val_impair=base_v["acc_forget"],
                   Af_val_final=rec_v["acc_forget"], Fr_val_base=base_v["macro_f1_retain"],
                   Fr_val_final=rec_v["macro_f1_retain"],
                   test_forget_base=base_t["acc_forget"], test_forget_impair=base_t["acc_forget"],
                   test_forget_final=rec_t["acc_forget"],
                   test_retain_acc_base=base_t["acc_retain"], test_retain_acc_impair=base_t["acc_retain"],
                   test_retain_acc_final=rec_t["acc_retain"],
                   test_retain_f1_final=rec_t["macro_f1_retain"], test_acc_full_final=rec_t["acc_full"])
        csv = Path(cfg.get("out_dir", "outputs")) / "results_adaptive.csv"
        df = pd.DataFrame([row])
        if csv.exists():                       # keep the column order of the existing file
            df = df[list(pd.read_csv(csv, nrows=0).columns)]
        df.to_csv(csv, mode="a", header=not csv.exists(), index=False)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()},
                         indent=1))


if __name__ == "__main__":
    main()