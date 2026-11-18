"""Final thesis table + figures (pandas/matplotlib only, no GPU needed)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config_loader import load_cfg

cfg = load_cfg()
out = Path(cfg.get("out_dir", "outputs"))
figs = out / "figs"
figs.mkdir(exist_ok=True)
TAGS = ["C1", "C2", "C4", "Cm"]
METHODS = {"fixed": "_fixed", "vol_sup": "_fixed_vol_sup1.0", "nonoise_sup": "_nonoise_sup1.0"}
LABEL = {"fixed": "Base-paper control", "vol_sup": "Vol cap + suppression",
         "nonoise_sup": "Suppression only"}

adp = pd.read_csv(out / "results_adaptive.csv").drop_duplicates("run_id", keep="last")
adp = adp[~adp.run_id.str.contains("smoke")]
ver = pd.read_csv(out / "verification.csv").drop_duplicates(["tag", "method", "seed"], keep="last")


def ms(x, nd=3):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return "-"
    return f"{x.mean():.{nd}f}" + (f"±{x.std(ddof=1):.{nd}f}" if len(x) > 1 else "")


def mean_sd(x):
    x = pd.Series(x).dropna()
    return (x.mean() if len(x) else np.nan), (x.std(ddof=1) if len(x) > 1 else 0.0)


rows, plot = [], []
for tag in TAGS:
    rt = out / "references" / f"retrain_{tag}" / "metrics.json"
    r = json.load(open(rt)) if rt.exists() else None
    rt_time = r["train_time_s"] if r else np.nan
    if r:
        v = ver[(ver.tag == tag) & (ver.method == "retrain")]
        rows.append(dict(set=tag, method="Retrain (gold)", n=1, forget_acc="0.000",
                         retain_acc=f"{r['acc_retain']:.3f}", retain_F1=f"{r['macro_f1_retain']:.3f}",
                         time_s=f"{rt_time:.0f}", pct_of_retrain="100%",
                         MIA_AUC=ms(v.mia_auc), KL_forget="0.00"))
    for m, suf in METHODS.items():
        a = adp[(adp.tag == tag) & (adp.suffix == suf)]
        if a.empty:
            continue
        v = ver[(ver.tag == tag) & (ver.method == m)]
        pct = f"{100 * a.total_time_s.mean() / rt_time:.0f}%" if rt_time == rt_time else "-"
        rows.append(dict(set=tag, method=LABEL[m], n=len(a), forget_acc=ms(a.test_forget_final),
                         retain_acc=ms(a.test_retain_acc_final), retain_F1=ms(a.test_retain_f1_final),
                         time_s=ms(a.total_time_s, 0), pct_of_retrain=pct,
                         MIA_AUC=ms(v.mia_auc), KL_forget=ms(v.KL_forget, 2)))
        f, fs = mean_sd(a.test_forget_final)
        mi, mis = mean_sd(v.mia_auc)
        plot.append(dict(tag=tag, method=m, forget=f, forget_sd=fs, mia=mi, mia_sd=mis,
                         pct=100 * a.total_time_s.mean() / rt_time if rt_time == rt_time else np.nan))

table = pd.DataFrame(rows)
table.to_csv(out / "final_table.csv", index=False)
with open(out / "final_table.md", "w") as fh:
    fh.write("| " + " | ".join(table.columns) + " |\n|" + "---|" * len(table.columns) + "\n")
    for _, r in table.iterrows():
        fh.write("| " + " | ".join(str(x) for x in r.values) + " |\n")
print(table.to_string(index=False))

P = pd.DataFrame(plot)


def grouped(ax, col, err, ylabel, title, hline=None):
    tags = [t for t in TAGS if t in set(P.tag)]
    w = 0.26
    for j, m in enumerate(METHODS):
        sub = P[P.method == m].set_index("tag").reindex(tags)
        ax.bar(np.arange(len(tags)) + (j - 1) * w, sub[col], w, yerr=sub[err] if err else None,
               label=LABEL[m], capsize=3)
    ax.set_xticks(range(len(tags)), tags)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if hline is not None:
        ax.axhline(hline, color="k", ls="--", lw=1)
    ax.legend(fontsize=8)


fig, ax = plt.subplots(figsize=(6, 3.8))
grouped(ax, "forget", "forget_sd", "accuracy on forgotten classes (0 = ideal)", "Forgetting")
fig.tight_layout(); fig.savefig(figs / "fig_forgetting.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3.8))
grouped(ax, "mia", "mia_sd", "membership-inference AUC (0.5 = chance)", "Privacy: membership inference", 0.5)
bl = ver[ver.method == "baseline"].groupby("tag").mia_auc.mean()
rtm = ver[ver.method == "retrain"].groupby("tag").mia_auc.mean()
tags = [t for t in TAGS if t in set(P.tag)]
for i, t in enumerate(tags):
    if t in bl:
        ax.plot(i, bl[t], "kx", ms=9, label="baseline" if i == 0 else None)
    if t in rtm:
        ax.plot(i, rtm[t], "kD", ms=6, mfc="none", label="retrain" if i == 0 else None)
ax.set_ylim(0.4, 0.9); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(figs / "fig_privacy.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3.8))
grouped(ax, "pct", None, "unlearning time, % of retrain-from-scratch", "Cost", 100)
fig.tight_layout(); fig.savefig(figs / "fig_cost.png", dpi=150); plt.close(fig)
print("Saved final_table.csv/.md and figs/fig_{forgetting,privacy,cost}.png to", out)