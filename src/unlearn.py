"""Base-paper (Patel et al. 2026 / Tarun et al.) single-shot impair-repair unlearning.

Usage (from repo root):
  python src/unlearn.py --baseline <out_dir>/baseline/best.pt --forget C1 C2 C4 Cm
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from config_loader import load_cfg
from dataset import (CLASSES, CLASS_TO_IDX, HAMDataset, balanced_sampler,
                     get_transforms)
from utils import atomic_save, build_model, evaluate, load_weights, set_seed

FORGET_SETS = {                      # paper Table 5
    "C1": ["bcc"],
    "C2": ["bkl", "akiec"],
    "C4": ["mel", "bkl", "bcc", "akiec"],
    "Cm": ["nv"],
}


class NoisyBank(Dataset):
    """Holds the noisy forget images (X_f + N) in fp16 on CPU to save RAM."""
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i].float(), int(self.y[i])


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


# ------------------------------------------------------------------ noise
def make_noise_bank(model, forget_ds, cfg, device, seed):
    """Step A: error-maximising noise.  loss = -CE(f(X+N), y) + lambda*||N||^2"""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(forget_ds, batch_size=cfg["batch_size"], shuffle=True,
                    num_workers=cfg["num_workers"], generator=g)
    xs, ys = [], []
    cap = cfg.get("max_noise_batches")
    for bi, (x, y) in enumerate(dl):
        if cap and bi >= cap:
            break
        x, y = x.to(device), y.to(device)
        N = (torch.randn_like(x) * cfg["noise_init_std"]).requires_grad_(True)
        opt = torch.optim.Adam([N], lr=cfg["noise_lr"])
        for _ in range(cfg["noise_steps"]):
            opt.zero_grad(set_to_none=True)
            ce = F.cross_entropy(model(x + N), y)
            reg = N.flatten(1).pow(2).sum(1).mean()
            loss = -ce + cfg["noise_lambda"] * reg
            loss.backward()
            opt.step()
        xs.append((x + N).detach().half().cpu())
        ys.append(y.cpu())
    for p in model.parameters():
        p.requires_grad_(True)
    return torch.cat(xs), torch.cat(ys)


# ------------------------------------------------------------ train epoch
def make_opt(name, params, lr, momentum=0.0):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum)
    return torch.optim.Adam(params, lr=lr)


def train_epochs(model, dl, opt, device, epochs, amp):
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    for _ in range(epochs):
        model.train()
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                loss = F.cross_entropy(model(x).float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()


# ---------------------------------------------------------------- metrics
def summarize(y, p, forget_idx):
    fm = np.isin(y, forget_idx)
    out = {
        "acc_full": float((y == p).mean()),
        "acc_forget": float((p[fm] == y[fm]).mean()) if fm.any() else float("nan"),
        "acc_retain": float((p[~fm] == y[~fm]).mean()) if (~fm).any() else float("nan"),
    }
    keep = [k for k in range(len(CLASSES)) if k not in forget_idx]
    out["macro_f1_retain"] = float(f1_score(y[~fm], p[~fm], labels=keep,
                                            average="macro", zero_division=0))
    out["recall"] = {c: (float((p[y == k] == k).mean()) if (y == k).any() else None)
                     for k, c in enumerate(CLASSES)}
    return out


def param_drift(m_new, m_old):
    s = 0.0
    for a, b in zip(m_new.parameters(), m_old.parameters()):
        s += (a.detach().float() - b.detach().float()).pow(2).sum().item()
    return s ** 0.5


# -------------------------------------------------------------------- run
def run_one(tag, cfg, args, device, amp, test_dl, train_csv_ds_eval):
    run_id = f"resnet50_{tag}_seed{args.seed}{args.suffix}"
    out_dir = Path(cfg.get("out_dir", "outputs")) / "unlearning" / run_id
    if (out_dir / "metrics.json").exists() and not args.force:
        print(f"[skip] {run_id} already done")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    forget_idx = [CLASS_TO_IDX[c] for c in FORGET_SETS[tag]]
    labels = np.array(train_csv_ds_eval.labels)
    f_ids = np.where(np.isin(labels, forget_idx))[0].tolist()
    r_ids = np.where(~np.isin(labels, forget_idx))[0].tolist()
    pct_forget = 100.0 * len(f_ids) / len(labels)
    print(f"\n=== {run_id}: forget {FORGET_SETS[tag]} "
          f"({len(f_ids)} train imgs, {pct_forget:.1f}%) ===")

    model = load_weights(build_model(pretrained=False), args.baseline, device).to(device)
    orig = copy.deepcopy(model).to(device).eval()

    y, p, _ = evaluate(model, test_dl, device, amp=amp)
    base = summarize(y, p, forget_idx)                      # A_i

    # Step A: noise
    t0 = time.time()
    xb, yb = make_noise_bank(model, Subset(train_csv_ds_eval, f_ids), cfg, device, args.seed)
    sync(device); t_noise = time.time() - t0
    print(f"noise bank: {len(yb)} samples in {t_noise:.0f}s")

    # Step B: impair  (noisy forget samples + a fraction of retain data)
    rng = np.random.RandomState(args.seed)
    n_keep = max(1, int(cfg["retain_fraction"] * len(r_ids)))
    r_sub = rng.choice(r_ids, size=n_keep, replace=False).tolist()
    impair_ds = ConcatDataset([NoisyBank(xb, yb), Subset(train_csv_ds_eval, r_sub)])
    impair_dl = DataLoader(impair_ds, batch_size=cfg["batch_size"], shuffle=True,
                           num_workers=cfg["num_workers"])
    sync(device); t0 = time.time()
    opt = make_opt(cfg["impair_optim"], model.parameters(), cfg["impair_lr"], cfg["impair_momentum"])
    train_epochs(model, impair_dl, opt, device, cfg["impair_epochs"], amp)
    sync(device); t_impair = time.time() - t0
    y, p, _ = evaluate(model, test_dl, device, amp=amp)
    perturbed = summarize(y, p, forget_idx)                 # A_p
    del xb, yb, impair_ds, impair_dl

    # Step C: repair  (retain data only)
    r_ds = Subset(train_csv_ds_eval, r_ids)
    if cfg["repair_balanced"]:
        sampler = balanced_sampler([labels[i] for i in r_ids])
        repair_dl = DataLoader(r_ds, batch_size=cfg["batch_size"], sampler=sampler,
                               num_workers=cfg["num_workers"])
    else:
        repair_dl = DataLoader(r_ds, batch_size=cfg["batch_size"], shuffle=True,
                               num_workers=cfg["num_workers"])
    sync(device); t0 = time.time()
    opt = make_opt(cfg["repair_optim"], model.parameters(), cfg["repair_lr"])
    train_epochs(model, repair_dl, opt, device, cfg["repair_epochs"], amp)
    sync(device); t_repair = time.time() - t0
    y, p, _ = evaluate(model, test_dl, device, amp=amp)
    recovered = summarize(y, p, forget_idx)                 # A_r

    res = {"run_id": run_id, "tag": tag, "forget_classes": FORGET_SETS[tag],
           "seed": args.seed, "pct_forget": pct_forget,
           "base": base, "perturbed": perturbed, "recovered": recovered,
           "noise_time_s": t_noise, "impair_time_s": t_impair, "repair_time_s": t_repair,
           "RT_s": t_impair + t_repair, "MPD": param_drift(model, orig),
           "cfg": {k: cfg[k] for k in cfg if k.startswith(("noise", "impair", "repair", "retain"))}}
    for key in ("acc_full", "acc_retain"):                  # both readings of "accuracy"
        Ai, Ap, Ar = base[key], perturbed[key], recovered[key]
        res[f"PID_{key}"] = Ai - Ap
        res[f"REQ_{key}"] = Ar / Ai
        res[f"UII_{key}"] = (Ai - Ar) / pct_forget
    json.dump(res, open(out_dir / "metrics.json", "w"), indent=2)
    atomic_save({"model": model.state_dict()}, out_dir / "final.pt")

    row = {k: res[k] for k in ["run_id", "tag", "seed", "pct_forget", "MPD", "RT_s",
                               "noise_time_s", "PID_acc_full", "REQ_acc_full", "UII_acc_full",
                               "PID_acc_retain", "REQ_acc_retain", "UII_acc_retain"]}
    row.update(Ai_full=base["acc_full"], Ap_full=perturbed["acc_full"], Ar_full=recovered["acc_full"],
               Ai_retain=base["acc_retain"], Ap_retain=perturbed["acc_retain"],
               Ar_retain=recovered["acc_retain"],
               forget_acc_before=base["acc_forget"], forget_acc_after=recovered["acc_forget"])
    csv = Path(cfg.get("out_dir", "outputs")) / "results.csv"
    pd.DataFrame([row]).to_csv(csv, mode="a", header=not csv.exists(), index=False)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--forget", nargs="+", default=["C1", "C2", "C4", "Cm"], choices=list(FORGET_SETS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suffix", default="", help="e.g. _rep2 for ablations")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="override config, e.g. repair_epochs=2")
    args = ap.parse_args()

    cfg = load_cfg()
    for kv in args.set:
        k, v = kv.split("=")
        cfg[k] = json.loads(v) if v not in ("sgd", "adam") else v
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = bool(cfg.get("use_amp", True)) and device == "cuda"

    d = cfg["splits_dir"]
    train_eval = HAMDataset(f"{d}/train.csv", cfg["data_root"], get_transforms(False, cfg["image_size"]))
    test_ds = HAMDataset(f"{d}/test.csv", cfg["data_root"], get_transforms(False, cfg["image_size"]))
    test_dl = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                         num_workers=cfg["num_workers"])
    for tag in args.forget:
        run_one(tag, cfg, args, device, amp, test_dl, train_eval)


if __name__ == "__main__":
    main()