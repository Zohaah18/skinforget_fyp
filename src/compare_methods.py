"""Compare baseline / control / fine-tune-only / retrain (+ best adaptive families) on the test set.
Gold standard = retrain-from-scratch."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from config_loader import load_cfg
from dataset import CLASSES, CLASS_TO_IDX, HAMDataset, get_transforms
from unlearn import FORGET_SETS
from utils import build_model, load_weights


@torch.no_grad()
def get_probs(path, dl, device, amp):
    model = load_weights(build_model(pretrained=False), path, device).to(device).eval()
    ys, ps = [], []
    for x, y in dl:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            out = model(x.to(device))
        ps.append(torch.softmax(out.float(), 1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def kl(p, q, eps=1e-8):
    p, q = np.clip(p, eps, 1.0), np.clip(q, eps, 1.0)
    return (p * (np.log(p) - np.log(q))).sum(1)      # per-sample KL(gold || method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--control_suffix", default="", help="suffix of the control run, e.g. '' or _amp")
    ap.add_argument("--forget", nargs="+", default=["C1", "C2", "C4", "Cm"], choices=list(FORGET_SETS))
    ap.add_argument("--best_json", default=None, help="adaptive_best.json: adds one method per family")
    ap.add_argument("--out_name", default="comparison", help="output file stem")
    args = ap.parse_args()

    cfg = load_cfg()
    out = Path(cfg.get("out_dir", "outputs"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = bool(cfg.get("use_amp", True)) and device == "cuda"
    d, sz = cfg["splits_dir"], cfg["image_size"]
    test_ds = HAMDataset(f"{d}/test.csv", cfg["data_root"], get_transforms(False, sz))
    dl = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                    num_workers=cfg["num_workers"])
    best = json.load(open(args.best_json)) if args.best_json else {}

    rows, recall_rows = [], []
    for tag in args.forget:
        fidx = [CLASS_TO_IDX[c] for c in FORGET_SETS[tag]]
        keep = [k for k in range(len(CLASSES)) if k not in fidx]
        paths = {
            "baseline": args.baseline,
            "control": out / "unlearning" / f"resnet50_{tag}_seed{args.seed}{args.control_suffix}" / "final.pt",
            "finetune_only": out / "references" / f"finetune_{tag}_seed{args.seed}" / "final.pt",
            "retrain": out / "references" / f"retrain_{tag}" / "best.pt",
        }
        for fam, per_tag in best.items():
            if tag in per_tag:
                paths[fam] = out / "unlearning" / f"resnet50_{tag}_seed{args.seed}{per_tag[tag]}" / "final.pt"
        probs, y = {}, None
        for name, pth in paths.items():
            if not Path(pth).exists():
                print(f"[missing] {tag} {name}: {pth}")
                continue
            y, probs[name] = get_probs(pth, dl, device, amp)
        if "retrain" not in probs:
            continue
        gold = probs["retrain"]
        gold_pred = gold.argmax(1)
        fm = np.isin(y, fidx)
        for name, pr in probs.items():
            pred = pr.argmax(1)
            k = kl(gold, pr)
            rows.append(dict(
                tag=tag, method=name,
                acc_full=(pred == y).mean(),
                acc_retain=(pred[~fm] == y[~fm]).mean(),
                acc_forget=(pred[fm] == y[fm]).mean(),
                macro_f1_retain=f1_score(y[~fm], pred[~fm], labels=keep,
                                         average="macro", zero_division=0),
                KL_full=k.mean(), KL_retain=k[~fm].mean(), KL_forget=k[fm].mean(),
                agree_full=(pred == gold_pred).mean(),
                agree_retain=(pred[~fm] == gold_pred[~fm]).mean(),
                agree_forget=(pred[fm] == gold_pred[fm]).mean()))
            recall_rows.append(dict(tag=tag, method=name, **{
                c: float((pred[y == i] == i).mean()) for i, c in enumerate(CLASSES)}))

    df = pd.DataFrame(rows).round(4)
    rec = pd.DataFrame(recall_rows).round(3)
    df.to_csv(out / f"{args.out_name}.csv", index=False)
    rec.to_csv(out / f"{args.out_name}_recall.csv", index=False)
    print(df.to_string(index=False))
    print(rec.to_string(index=False))
    print(f"Saved {args.out_name}.csv and {args.out_name}_recall.csv to", out)


if __name__ == "__main__":
    main()