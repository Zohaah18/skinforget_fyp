import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from torch.utils.data import DataLoader, Subset

from config_loader import load_cfg
from dataset import get_loaders, class_weights, CLASSES, CLASS_TO_IDX
from unlearn import FORGET_SETS
from utils import set_seed, build_model, atomic_save, load_weights, evaluate


def retain_loader(ds, forget_idx, cfg, shuffle):
    ids = [i for i, l in enumerate(ds.labels) if l not in forget_idx]
    return DataLoader(Subset(ds, ids), batch_size=cfg["batch_size"], shuffle=shuffle,
                      num_workers=cfg["num_workers"], pin_memory=torch.cuda.is_available())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--max_batches", type=int, default=None, help="smoke tests only")
    ap.add_argument("--run_name", default="baseline")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--fresh", action="store_true", help="ignore existing last.pt")
    ap.add_argument("--forget", default=None, choices=list(FORGET_SETS),
                    help="train a retrain-from-scratch reference on the retain set only")
    args = ap.parse_args()

    cfg = load_cfg()
    seed = cfg["seed"] if args.seed is None else args.seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = bool(cfg.get("use_amp", True)) and device == "cuda"
    run_name = f"references/retrain_{args.forget}" if args.forget else args.run_name
    if args.max_batches:
        run_name += "_smoke"             # smoke tests never collide with real runs
    out = Path(cfg.get("out_dir", "outputs")) / run_name
    out.mkdir(parents=True, exist_ok=True)
    print("Device:", device, "| AMP:", amp, "| out:", out)

    set_seed(seed)
    train_dl, val_dl, test_dl, train_ds = get_loaders(cfg)
    forget_idx = [CLASS_TO_IDX[c] for c in FORGET_SETS[args.forget]] if args.forget else []
    f1_labels = [k for k in range(len(CLASSES)) if k not in forget_idx]
    if forget_idx:
        train_dl = retain_loader(train_ds, forget_idx, cfg, True)
        val_dl = retain_loader(val_dl.dataset, forget_idx, cfg, False)
        print(f"RETRAIN reference: forgetting {FORGET_SETS[args.forget]}, "
              f"{len(train_dl.dataset)} train images remain")

    model = build_model().to(device)
    weight = class_weights(train_ds.labels).to(device) if cfg.get("use_class_weights") else None
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_ep, best_f1, bad, history = 0, -1.0, 0, []
    last_path, best_path = out / "last.pt", out / "best.pt"
    if last_path.exists() and not args.fresh:
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_ep, best_f1, bad, history = ck["epoch"], ck["best_f1"], ck["bad"], ck["history"]
        print(f"RESUMED from epoch {start_ep} (best val F1 so far {best_f1:.4f})")

    for ep in range(start_ep, args.epochs):
        if bad >= args.patience:
            break
        set_seed(seed + ep)          # makes a resumed run follow the same path
        model.train()
        t0, run, n = time.time(), 0.0, 0
        for i, (x, y) in enumerate(train_dl):
            if args.max_batches and i >= args.max_batches:
                break
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                loss = loss_fn(model(x).float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item() * len(y)
            n += len(y)
            if i % 50 == 0:
                print(f"  batch {i}  loss {loss.item():.3f}", flush=True)
        sched.step()

        y, p, vloss = evaluate(model, val_dl, device, loss_fn, args.max_batches, amp)
        f1 = float(f1_score(y, p, average="macro", labels=f1_labels, zero_division=0))
        vacc = float(accuracy_score(y, p))
        history.append(dict(epoch=ep + 1, train_loss=run / n, val_loss=vloss,
                            val_macro_f1=f1, val_acc=vacc, secs=time.time() - t0))
        print(f"epoch {ep+1}/{args.epochs}  train_loss {run/n:.4f}  val_loss {vloss:.4f}  "
              f"val_acc {vacc:.4f}  val_macroF1 {f1:.4f}  {time.time()-t0:.0f}s", flush=True)

        if f1 > best_f1:
            best_f1, bad = f1, 0
            atomic_save({"model": model.state_dict(), "epoch": ep + 1, "val_f1": f1}, best_path)
        else:
            bad += 1
        atomic_save({"model": model.state_dict(), "opt": opt.state_dict(),
                     "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                     "epoch": ep + 1, "best_f1": best_f1, "bad": bad, "history": history,
                     "seed": seed}, last_path)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        if bad >= args.patience:
            print("Early stopping")

    # ---- final test evaluation with the best checkpoint (full test set) ----
    load_weights(model, best_path, device)
    y, p, _ = evaluate(model, test_dl, device, None, args.max_batches, amp)
    labels = list(range(len(CLASSES)))
    keep_mask = ~np.isin(y, forget_idx)
    metrics = {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "acc_retain": float((y[keep_mask] == p[keep_mask]).mean()),
        "macro_f1_retain": float(f1_score(y[keep_mask], p[keep_mask], labels=f1_labels,
                                          average="macro", zero_division=0)),
        "per_class_recall": {c: float((p[y == k] == k).mean()) if (y == k).any() else None
                             for c, k in zip(CLASSES, labels)},
        "train_time_s": float(sum(h["secs"] for h in history)),
        "epochs_run": len(history),
        "seed": seed, "forget": args.forget,
        "recipe": {k: cfg[k] for k in
                   ["use_balanced_sampler", "use_class_weights", "use_augmentation"]},
    }
    print(metrics)
    print(classification_report(y, p, labels=labels, target_names=CLASSES, zero_division=0))
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)

    cm = confusion_matrix(y, p, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(labels, CLASSES, rotation=45); ax.set_yticks(labels, CLASSES)
    for i in labels:
        for j in labels:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=150)
    print("Saved metrics + confusion matrix to", out)


if __name__ == "__main__":
    main()