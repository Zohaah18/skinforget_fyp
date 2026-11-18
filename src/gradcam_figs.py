"""Grad-CAM comparison on forgotten-class test images: baseline vs unlearned models vs retrain.

  python src/gradcam_figs.py --baseline B --tag C1 --n 6
Needs:  pip install grad-cam
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from config_loader import load_cfg
from dataset import CLASS_TO_IDX, CLASSES, MEAN, STD
from unlearn import FORGET_SETS
from utils import build_model, load_weights
from verify import load_cache, model_path

METHODS = [("baseline", "baseline"), ("control_v2", "base-paper\ncontrol"),
           ("vol_sup", "vol cap +\nsuppression"), ("nonoise_sup", "suppression\nonly"),
           ("retrain", "retrain\n(gold)")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--tag", default="C1", choices=list(FORGET_SETS))
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", default="true", choices=["true", "pred"],
                    help="true: CAM for the forgotten (true) class; pred: CAM for each model's own prediction")
    args = ap.parse_args()

    cfg = load_cfg()
    out = Path(cfg.get("out_dir", "outputs"))
    (out / "figs").mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xte, yte = load_cache(cfg)["test"]
    fidx = [CLASS_TO_IDX[c] for c in FORGET_SETS[args.tag]]
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)

    def prep(i):
        return ((Xte[i].float().unsqueeze(0) / 255.0 - mean) / std).to(device)

    models, cams = {}, {}
    for m, _ in METHODS:
        mdl = load_weights(build_model(pretrained=False),
                           model_path(m, out, args.tag, args.seed, args.baseline), device).to(device).eval()
        models[m] = mdl
        cams[m] = GradCAM(model=mdl, target_layers=[mdl.layer4[-1]])

    # forgotten-class test images that the baseline classifies correctly
    rng = np.random.RandomState(0)
    cand = rng.permutation(np.where(np.isin(yte, fidx))[0])
    sel = []
    for i in cand:
        with torch.no_grad():
            if int(models["baseline"](prep(i)).argmax(1)) == int(yte[i]):
                sel.append(int(i))
        if len(sel) == args.n:
            break

    fig, axes = plt.subplots(len(sel), 1 + len(METHODS), figsize=(2.6 * (1 + len(METHODS)), 2.7 * len(sel)))
    axes = np.atleast_2d(axes)
    corr = {m: [] for m, _ in METHODS}
    for r, i in enumerate(sel):
        y = int(yte[i])
        rgb = (Xte[i].permute(1, 2, 0).numpy() / 255.0).astype(np.float32)
        axes[r, 0].imshow(rgb)
        axes[r, 0].set_title(f"true: {CLASSES[y]}", fontsize=9)
        x = prep(i)
        base_map = None
        for c, (m, label) in enumerate(METHODS, start=1):
            with torch.no_grad():
                p = torch.softmax(models[m](x).float(), 1)[0].cpu().numpy()
            pred = int(p.argmax())
            tgt = y if args.target == "true" else pred
            g = cams[m](input_tensor=x, targets=[ClassifierOutputTarget(tgt)])[0]
            if base_map is None:
                base_map = g
            corr[m].append(float(np.corrcoef(g.ravel(), base_map.ravel())[0, 1]))
            axes[r, c].imshow(show_cam_on_image(rgb, g, use_rgb=True))
            axes[r, c].set_title(f"{label if r == 0 else ''}\npred {CLASSES[pred]}  p(true)={p[y]:.2f}",
                                 fontsize=8)
    for a in axes.ravel():
        a.axis("off")
    fig.suptitle(f"Grad-CAM on forgotten-class images ({FORGET_SETS[args.tag]}), "
                 f"target = {args.target} class", fontsize=11)
    fig.tight_layout()
    path = out / "figs" / f"gradcam_{args.tag}_{args.target}.png"
    fig.savefig(path, dpi=130)
    print("Saved", path)
    print("Mean correlation of each model's CAM with the baseline CAM:")
    for m, _ in METHODS:
        print(f"  {m:12s} {np.nanmean(corr[m]):.3f}")


if __name__ == "__main__":
    main()