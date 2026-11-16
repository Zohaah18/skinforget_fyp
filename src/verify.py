"""Verification suite: membership inference, linear probe, and KL/agreement vs the retrain gold model.

  python src/verify.py --baseline B --forget C1 C2 Cm --seeds 0 1 2
Resumable: appends one row per (tag, method, seed) to <out_dir>/<out_name>.csv and skips finished rows.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from config_loader import load_cfg
from dataset import CLASS_TO_IDX, CLASSES, HAMDataset, MEAN, STD
from unlearn import FORGET_SETS
from utils import build_model, load_weights
from verify_metrics import kl_to_gold, linear_probe, log_softmax, mia, sample_stats

ALL_METHODS = ["baseline", "retrain", "finetune_only", "control_v2", "fixed", "vol_sup", "nonoise_sup"]
SEEDED = {"fixed", "vol_sup", "nonoise_sup"}


def model_path(method, out, tag, seed, baseline):
    u = out / "unlearning"
    return {
        "baseline": Path(baseline),
        "retrain": out / "references" / f"retrain_{tag}" / "best.pt",
        "finetune_only": out / "references" / f"finetune_{tag}_seed0" / "final.pt",
        "control_v2": u / f"resnet50_{tag}_seed0" / "final.pt",
        "fixed": u / f"resnet50_{tag}_seed{seed}_fixed" / "final.pt",
        "vol_sup": u / f"resnet50_{tag}_seed{seed}_fixed_vol_sup1.0" / "final.pt",
        "nonoise_sup": u / f"resnet50_{tag}_seed{seed}_nonoise_sup1.0" / "final.pt",
    }[method]


def load_cache(cfg):
    """Decode every image once (uint8, 224x224) so each model pass takes seconds, not minutes."""
    path = Path(cfg.get("cache_path", "image_cache.pt"))
    if path.exists():
        return torch.load(path, weights_only=False)
    d, sz = cfg["splits_dir"], cfg["image_size"]
    tf = T.Compose([T.Resize((sz, sz)), T.PILToTensor()])
    data = {}
    for split in ["train", "val", "test"]:
        print("caching", split, flush=True)
        dl = DataLoader(HAMDataset(f"{d}/{split}.csv", cfg["data_root"], tf), batch_size=64,
                        shuffle=False, num_workers=max(cfg["num_workers"], 2))
        xs, ys = [], []
        for x, y in dl:
            xs.append(x)
            ys.append(y)
        data[split] = (torch.cat(xs), torch.cat(ys).numpy())
    torch.save(data, path)
    return data


@torch.no_grad()
def forward_all(model, X, device, amp, bs=128):
    """Returns (penultimate 2048-d features, logits) for a uint8 image tensor."""
    model.eval()
    body = torch.nn.Sequential(*list(model.children())[:-1])
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    feats, logits = [], []
    for i in range(0, len(X), bs):
        x = (X[i:i + bs].to(device).float() / 255.0 - mean) / std
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            f = body(x).flatten(1)
            z = model.fc(f)
        feats.append(f.float().cpu())
        logits.append(z.float().cpu())
    return torch.cat(feats).numpy(), torch.cat(logits).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--forget", nargs="+", default=["C1", "C2", "Cm"], choices=list(FORGET_SETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    ap.add_argument("--out_name", default="verification")
    args = ap.parse_args()

    cfg = load_cfg()
    out = Path(cfg.get("out_dir", "outputs"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = bool(cfg.get("use_amp", True)) and device == "cuda"
    data = load_cache(cfg)
    ytr, yva, yte = data["train"][1], data["val"][1], data["test"][1]
    csv = out / f"{args.out_name}.csv"
    done = set()
    if csv.exists():
        old = pd.read_csv(csv)
        done = set(zip(old.tag, old.method, old.seed))

    def run_model(path):
        model = load_weights(build_model(pretrained=False), path, device).to(device)
        res = {s: forward_all(model, data[s][0], device, amp) for s in ["train", "val", "test"]}
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        return res

    for tag in args.forget:
        fidx = [CLASS_TO_IDX[c] for c in FORGET_SETS[tag]]
        gold_path = model_path("retrain", out, tag, 0, args.baseline)
        if not gold_path.exists():
            print(f"[missing] retrain for {tag}: {gold_path}")
            continue
        gold = run_model(gold_path)
        gold_p = np.exp(log_softmax(gold["test"][1].astype(np.float64)))
        gpred = gold_p.argmax(1)
        mem = np.where(np.isin(ytr, fidx))[0]                  # members: forgotten TRAIN images
        non_val = np.where(np.isin(yva, fidx))[0]              # non-members: same classes, unseen
        non_te = np.where(np.isin(yte, fidx))[0]
        fm_te = np.isin(yte, fidx)

        for method in args.methods:
            for seed in (args.seeds if method in SEEDED else [0]):
                if (tag, method, seed) in done:
                    continue
                path = model_path(method, out, tag, seed, args.baseline)
                if not path.exists():
                    print(f"[missing] {tag} {method} seed{seed}: {path}")
                    continue
                t0 = time.time()
                r = gold if method == "retrain" else run_model(path)
                (Ftr, Ltr), (Fva, Lva), (Fte, Lte) = r["train"], r["val"], r["test"]
                st_m = sample_stats(Ltr[mem], ytr[mem])
                st_v = sample_stats(Lva[non_val], yva[non_val])
                st_t = sample_stats(Lte[non_te], yte[non_te])
                st_n = tuple(np.concatenate([a, b]) for a, b in zip(st_v, st_t))
                m = mia(st_m, st_n, seed=seed)
                pr = linear_probe(Ftr, np.isin(ytr, fidx).astype(int), Fte, fm_te.astype(int))
                p_te = np.exp(log_softmax(Lte.astype(np.float64)))
                kl = kl_to_gold(gold_p, p_te)
                pred = Lte.argmax(1)

                def pmass(L):                                   # mean probability on forgotten classes
                    return float(np.exp(log_softmax(L.astype(np.float64)))[:, fidx].sum(1).mean())

                row = dict(
                    tag=tag, method=method, seed=seed, n_member=len(mem),
                    n_nonmember=len(non_val) + len(non_te),
                    mia_auc=m["auc_lr"], mia_tpr1=m["tpr1_lr"], mia_auc_loss=m["auc_loss"],
                    mia_auc_conf=m["auc_conf"], mia_auc_ent=m["auc_ent"],
                    probe_bal_acc=pr["bal_acc"], probe_auc=pr["auc"],
                    pmass_member=pmass(Ltr[mem]),
                    pmass_nonmember=pmass(np.vstack([Lva[non_val], Lte[non_te]])),
                    acc_forget=float((pred[fm_te] == yte[fm_te]).mean()),
                    acc_retain=float((pred[~fm_te] == yte[~fm_te]).mean()),
                    KL_forget=float(kl[fm_te].mean()), KL_retain=float(kl[~fm_te].mean()),
                    agree_forget=float((pred[fm_te] == gpred[fm_te]).mean()),
                    agree_retain=float((pred[~fm_te] == gpred[~fm_te]).mean()))
                pd.DataFrame([row]).to_csv(csv, mode="a", header=not csv.exists(), index=False)
                print(f"{tag} {method:14s} seed{seed}  MIA-AUC {row['mia_auc']:.3f}  "
                      f"TPR@1%FPR {row['mia_tpr1']:.3f}  probe-bal-acc {row['probe_bal_acc']:.3f}  "
                      f"p(forgotten) members {row['pmass_member']:.4f} / non-members "
                      f"{row['pmass_nonmember']:.4f}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()