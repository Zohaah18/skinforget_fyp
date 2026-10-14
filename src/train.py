import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, accuracy_score)
from torchvision import models

from dataset import get_loaders, class_weights, CLASSES


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_model():
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    return m


@torch.no_grad()
def evaluate(model, dl, device, loss_fn=None, max_batches=None):
    model.eval()
    ys, ps, total, n = [], [], 0.0, 0
    for i, (x, y) in enumerate(dl):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        out = model(x)
        if loss_fn is not None:
            total += loss_fn(out, y).item() * len(y)
        n += len(y)
        ps.append(out.argmax(1).cpu())
        ys.append(y.cpu())
    return torch.cat(ys).numpy(), torch.cat(ps).numpy(), total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--max_batches", type=int, default=None,
                    help="limit batches per epoch (for quick smoke tests)")
    ap.add_argument("--out", default="checkpoints/baseline.pth")
    args = ap.parse_args()

    cfg = yaml.safe_load(open("configs/config.yaml"))
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_dl, val_dl, test_dl, train_ds = get_loaders(cfg)
    model = build_model().to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_ds.labels).to(device))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best_f1, bad = -1.0, 0

    for ep in range(args.epochs):
        model.train()
        t0, run, n = time.time(), 0.0, 0
        for i, (x, y) in enumerate(train_dl):
            if args.max_batches and i >= args.max_batches:
                break
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            run += loss.item() * len(y)
            n += len(y)
            if i % 10 == 0:
                print(f"  batch {i}  loss {loss.item():.3f}", flush=True)
        sched.step()

        y, p, vloss = evaluate(model, val_dl, device, loss_fn, args.max_batches)
        f1 = f1_score(y, p, average="macro")
        print(f"epoch {ep+1}/{args.epochs}  train_loss {run/n:.4f}  "
              f"val_loss {vloss:.4f}  val_macroF1 {f1:.4f}  {time.time()-t0:.0f}s")

        if f1 > best_f1:
            best_f1, bad = f1, 0
            torch.save(model.state_dict(), args.out)
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping")
                break

    # Final test evaluation with the best checkpoint
    model.load_state_dict(torch.load(args.out, map_location=device))
    y, p, _ = evaluate(model, test_dl, device, None, args.max_batches)
    metrics = {
        "accuracy": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y, p),
    }
    print(metrics)
    print(classification_report(y, p, labels=list(range(len(CLASSES))),
                                target_names=CLASSES, zero_division=0))

    Path("results").mkdir(exist_ok=True)
    json.dump(metrics, open("results/baseline_metrics.json", "w"), indent=2)
    cm = confusion_matrix(y, p, labels=list(range(len(CLASSES))))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig("results/confusion_matrix.png", dpi=150)
    print("Saved results/baseline_metrics.json and results/confusion_matrix.png")


if __name__ == "__main__":
    main()