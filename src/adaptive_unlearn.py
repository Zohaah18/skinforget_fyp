"""Adaptive impair-repair unlearning (SkinForget contribution 1) + optional Stage-2 refinements.

Examples (repo root):
  # Stage 1 grid: alpha, lambda, beta scheduled by forget fraction rho
  python src/adaptive_unlearn.py --baseline B --forget C2 --gamma 0.5 1 2 --kappa 1 2 4
  # control re-run with this runner (sanity check + its validation score J)
  python src/adaptive_unlearn.py --baseline B --forget C2 --fixed
  # Stage 2 switches (work with --fixed or with a gamma/kappa pair)
  python src/adaptive_unlearn.py --baseline B --forget Cm --fixed --vol --suppress 1.0
"""
import argparse
import copy
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset

from config_loader import load_cfg
from dataset import CLASS_TO_IDX, CLASSES, HAMDataset, balanced_sampler, get_transforms
from unlearn import (FORGET_SETS, NoisyBank, make_noise_bank, make_opt,
                     param_drift, summarize, sync, train_epochs)
from utils import atomic_save, build_model, evaluate, load_weights, set_seed


# ------------------------------------------------------------ schedule + J
def schedule(rho, gamma, kappa, cfg):
    """alpha(rho)=alpha0(1-rho)^gamma, lambda(rho)=lambda0(1+kappa*rho), beta(rho)=beta0(1-rho)^gamma"""
    alpha = cfg["noise_lr"] * (1.0 - rho) ** gamma
    lam = cfg["noise_lambda"] * (1.0 + kappa * rho)
    beta = cfg["impair_lr"] * (1.0 - rho) ** gamma
    return alpha, lam, beta


def objective(base_v, new_v, mpd, cfg):
    """Drift-constrained objective J, computed on the VALIDATION split."""
    af0, af1 = base_v["acc_forget"], new_v["acc_forget"]
    fr0, fr1 = base_v["macro_f1_retain"], new_v["macro_f1_retain"]
    t_forget = (af0 - af1) / max(af0, 1e-8)
    p_retain = cfg["mu"] * max(0.0, (fr0 - fr1) / max(fr0, 1e-8))
    p_mpd = cfg["nu"] * max(0.0, (mpd - cfg["mpd_budget"]) / cfg["mpd_budget"])
    return t_forget - p_retain - p_mpd, t_forget, p_retain, p_mpd


# ------------------------------------------------------------ repair step
def suppress_loss(logits, keep_t):
    """-log(1 - P(forgotten classes)), written with logsumexp so it never saturates."""
    logits = logits.float()
    return (torch.logsumexp(logits, 1) - torch.logsumexp(logits[:, keep_t], 1)).mean()


def cycle(dl):
    while True:
        for b in dl:
            yield b


def repair_epoch(model, retain_dl, forget_iter, opt, scaler, device, amp, keep_t, eta):
    model.train()
    for x, y in retain_dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            if forget_iter is not None:
                xf, _ = next(forget_iter)
                out = model(torch.cat([x, xf.to(device)]))
                loss = (F.cross_entropy(out[: len(x)].float(), y)
                        + eta * suppress_loss(out[len(x):], keep_t))
            else:
                loss = F.cross_entropy(model(x).float(), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()


# -------------------------------------------------------------------- run
def run_one(tag, gamma, kappa, cfg, args, device, amp, train_eval, val_dl, test_dl):
    fixed = args.fixed
    family = ("fixed" if fixed else "adapt") + ("+vol" if args.vol else "") + \
             ("+sup" if args.suppress > 0 else "")
    cfg_id = "fixed" if fixed else f"adapt_g{gamma}_k{kappa}"
    suffix = (f"_{cfg_id}" + ("_vol" if args.vol else "")
              + (f"_sup{args.suppress}" if args.suppress > 0 else "") + args.note)
    run_id = f"resnet50_{tag}_seed{args.seed}{suffix}"
    out_dir = Path(cfg.get("out_dir", "outputs")) / "unlearning" / run_id
    if (out_dir / "metrics.json").exists() and not args.force:
        print(f"[skip] {run_id} already done")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    forget_idx = [CLASS_TO_IDX[c] for c in FORGET_SETS[tag]]
    keep_idx = [k for k in range(len(CLASSES)) if k not in forget_idx]
    labels = np.array(train_eval.labels)
    f_ids = np.where(np.isin(labels, forget_idx))[0].tolist()
    r_ids = np.where(~np.isin(labels, forget_idx))[0].tolist()
    rho = len(f_ids) / len(labels)
    g_, k_ = (0.0, 0.0) if fixed else (gamma, kappa)
    alpha, lam, beta = schedule(rho, g_, k_, cfg)
    print(f"\n=== {run_id}: forget {FORGET_SETS[tag]}  rho={rho:.3f}  "
          f"alpha={alpha:.4f} lambda={lam:.5f} beta={beta:.4f} "
          f"vol={args.vol} suppress={args.suppress} ===")

    rcfg = dict(cfg)
    rcfg["noise_lr"], rcfg["noise_lambda"] = alpha, lam
    if args.vol:
        rcfg["max_noise_batches"] = int(np.ceil(cfg["vol_max_noisy"] / cfg["batch_size"]))

    model = load_weights(build_model(pretrained=False), args.baseline, device).to(device)
    orig = copy.deepcopy(model).to(device).eval()
    y, p, _ = evaluate(model, test_dl, device, amp=amp)
    base_t = summarize(y, p, forget_idx)
    y, p, _ = evaluate(model, val_dl, device, amp=amp)
    base_v = summarize(y, p, forget_idx)

    # ---- A: noise
    t0 = time.time()
    xb, yb = make_noise_bank(model, Subset(train_eval, f_ids), rcfg, device, args.seed)
    sync(device)
    t_noise = time.time() - t0
    n_noisy = len(yb)

    # ---- B: impair
    rng = np.random.RandomState(args.seed)
    n_keep = int(cfg["retain_fraction"] * len(r_ids))
    if args.vol:                                   # retain share >= noisy share
        n_keep = max(n_keep, n_noisy)
    n_keep = min(max(1, n_keep), len(r_ids))
    r_sub = rng.choice(r_ids, size=n_keep, replace=False).tolist()
    impair_dl = DataLoader(ConcatDataset([NoisyBank(xb, yb), Subset(train_eval, r_sub)]),
                           batch_size=cfg["batch_size"], shuffle=True,
                           num_workers=cfg["num_workers"])
    sync(device)
    t0 = time.time()
    opt = make_opt(cfg["impair_optim"], model.parameters(), beta, cfg["impair_momentum"])
    train_epochs(model, impair_dl, opt, device, cfg["impair_epochs"], amp)
    sync(device)
    t_impair = time.time() - t0
    y, p, _ = evaluate(model, test_dl, device, amp=amp)
    pert_t = summarize(y, p, forget_idx)
    y, p, _ = evaluate(model, val_dl, device, amp=amp)
    pert_v = summarize(y, p, forget_idx)
    del xb, yb, impair_dl

    # ---- C: repair (adaptive: up to repair_max_epochs, stop once val retain F1 is within 1pp)
    r_ds = Subset(train_eval, r_ids)
    if cfg["repair_balanced"]:
        retain_dl = DataLoader(r_ds, batch_size=cfg["batch_size"],
                               sampler=balanced_sampler([labels[i] for i in r_ids]),
                               num_workers=cfg["num_workers"])
    else:
        retain_dl = DataLoader(r_ds, batch_size=cfg["batch_size"], shuffle=True,
                               num_workers=cfg["num_workers"])
    forget_iter = None
    if args.suppress > 0:
        forget_iter = cycle(DataLoader(Subset(train_eval, f_ids), batch_size=cfg["batch_size"],
                                       shuffle=True, num_workers=cfg["num_workers"]))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    opt = make_opt(cfg["repair_optim"], model.parameters(), cfg["repair_lr"])
    keep_t = torch.tensor(keep_idx, device=device)
    max_ep = int(cfg["repair_epochs"]) if fixed else int(cfg["repair_max_epochs"])
    log = [{"epoch": 0, "val_acc_forget": pert_v["acc_forget"],
            "val_f1_retain": pert_v["macro_f1_retain"]}]
    t_repair, rec_v = 0.0, pert_v
    for ep in range(1, max_ep + 1):
        sync(device)
        t0 = time.time()
        repair_epoch(model, retain_dl, forget_iter, opt, scaler, device, amp, keep_t, args.suppress)
        sync(device)
        t_repair += time.time() - t0
        y, p, _ = evaluate(model, val_dl, device, amp=amp)
        rec_v = summarize(y, p, forget_idx)
        log.append({"epoch": ep, "val_acc_forget": rec_v["acc_forget"],
                    "val_f1_retain": rec_v["macro_f1_retain"]})
        if (not fixed) and rec_v["macro_f1_retain"] >= base_v["macro_f1_retain"] - 0.01:
            break
    y, p, _ = evaluate(model, test_dl, device, amp=amp)
    rec_t = summarize(y, p, forget_idx)

    mpd = param_drift(model, orig)
    J, j_f, j_r, j_m = objective(base_v, rec_v, mpd, cfg)
    res = {"run_id": run_id, "tag": tag, "family": family, "gamma": g_, "kappa": k_,
           "rho": rho, "alpha": alpha, "lambda": lam, "beta": beta, "vol": args.vol,
           "suppress_eta": args.suppress, "n_noisy": n_noisy, "n_retain_impair": n_keep,
           "J_val": J, "J_terms": [j_f, j_r, j_m], "MPD": mpd,
           "times": {"noise": t_noise, "impair": t_impair, "repair": t_repair},
           "repair_log": log, "val": {"base": base_v, "impair": pert_v, "final": rec_v},
           "test": {"base": base_t, "impair": pert_t, "final": rec_t}}
    json.dump(res, open(out_dir / "metrics.json", "w"), indent=2)
    atomic_save({"model": model.state_dict()}, out_dir / "final.pt")

    row = dict(run_id=run_id, suffix=suffix, tag=tag, family=family, gamma=g_, kappa=k_,
               rho=rho, alpha=alpha, lam=lam, beta=beta, vol=args.vol, eta=args.suppress,
               n_noisy=n_noisy, repair_epochs_used=len(log) - 1,
               J_val=J, J_forget=j_f, J_retain_pen=j_r, J_mpd_pen=j_m, MPD=mpd,
               noise_time_s=t_noise, impair_time_s=t_impair, repair_time_s=t_repair,
               total_time_s=t_noise + t_impair + t_repair,
               Af_val_base=base_v["acc_forget"], Af_val_impair=pert_v["acc_forget"],
               Af_val_final=rec_v["acc_forget"], Fr_val_base=base_v["macro_f1_retain"],
               Fr_val_final=rec_v["macro_f1_retain"],
               test_forget_base=base_t["acc_forget"], test_forget_impair=pert_t["acc_forget"],
               test_forget_final=rec_t["acc_forget"],
               test_retain_acc_base=base_t["acc_retain"], test_retain_acc_impair=pert_t["acc_retain"],
               test_retain_acc_final=rec_t["acc_retain"],
               test_retain_f1_final=rec_t["macro_f1_retain"], test_acc_full_final=rec_t["acc_full"])
    csv = Path(cfg.get("out_dir", "outputs")) / "results_adaptive.csv"
    pd.DataFrame([row]).to_csv(csv, mode="a", header=not csv.exists(), index=False)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()},
                     indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--forget", nargs="+", default=["C2", "Cm"], choices=list(FORGET_SETS))
    ap.add_argument("--gamma", nargs="+", type=float, default=[1.0])
    ap.add_argument("--kappa", nargs="+", type=float, default=[2.0])
    ap.add_argument("--fixed", action="store_true", help="no schedule: control hyperparameters")
    ap.add_argument("--vol", action="store_true", help="Stage 2: cap noise + balance impair mix")
    ap.add_argument("--suppress", type=float, default=0.0, help="Stage 2: forget-suppression weight eta")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--note", default="", help="appended to run id, e.g. _smoke")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="override config, e.g. noise_steps=5")
    args = ap.parse_args()

    cfg = load_cfg()
    for kv in args.set:
        k, v = kv.split("=")
        cfg[k] = json.loads(v) if v not in ("sgd", "adam") else v
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = bool(cfg.get("use_amp", True)) and device == "cuda"

    d, sz = cfg["splits_dir"], cfg["image_size"]
    tf = get_transforms(False, sz)
    train_eval = HAMDataset(f"{d}/train.csv", cfg["data_root"], tf)
    kw = dict(batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    val_dl = DataLoader(HAMDataset(f"{d}/val.csv", cfg["data_root"], tf), **kw)
    test_dl = DataLoader(HAMDataset(f"{d}/test.csv", cfg["data_root"], tf), **kw)

    for tag in args.forget:
        grid = [(0.0, 0.0)] if args.fixed else list(itertools.product(args.gamma, args.kappa))
        for g, k in grid:
            run_one(tag, g, k, cfg, args, device, amp, train_eval, val_dl, test_dl)


if __name__ == "__main__":
    main()