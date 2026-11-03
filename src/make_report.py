"""Builds the paper-style Table 6 and Figures 2-4 from outputs/results.csv."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config_loader import load_cfg

cfg = load_cfg()
out = Path(cfg.get("out_dir", "outputs"))
df = pd.read_csv(out / "results.csv").drop_duplicates("run_id", keep="last")
df = df[df.run_id.str.endswith(tuple(f"seed{s}" for s in range(10)))]   # main runs only (no ablations)
order = {"C1": 0, "C2": 1, "C4": 2, "Cm": 3}
df = df.sort_values("tag", key=lambda s: s.map(order))

cols = ["tag", "pct_forget", "Ai_full", "Ap_full", "Ar_full", "PID_acc_full", "REQ_acc_full",
        "Ai_retain", "Ap_retain", "Ar_retain", "PID_acc_retain", "REQ_acc_retain",
        "UII_acc_full", "RT_s", "MPD", "forget_acc_before", "forget_acc_after"]
table = df[cols].round(4)
table.to_csv(out / "table6_reproduction.csv", index=False)
print(table.to_string(index=False))

for col, title, fn in [("UII_acc_full", "Unlearning Impact Index (UII)", "fig2_uii.png"),
                       ("RT_s", "Relearn time (s)", "fig3_rt.png"),
                       ("MPD", "Model parameter drift (MPD)", "fig4_mpd.png")]:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(df.tag, df[col])
    ax.set_title(title); ax.set_xlabel("Unlearned class set")
    fig.tight_layout(); fig.savefig(out / fn, dpi=150)
print("Saved table and figures to", out)