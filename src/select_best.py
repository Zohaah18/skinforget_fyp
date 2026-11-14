"""Pick the best config per (forget set, family) by validation objective J; report TEST metrics."""
import json
from pathlib import Path

import pandas as pd

from config_loader import load_cfg

cfg = load_cfg()
out = Path(cfg.get("out_dir", "outputs"))
df = pd.read_csv(out / "results_adaptive.csv").drop_duplicates("run_id", keep="last")
df = df[~df.run_id.str.contains("smoke")]
df = df[df.run_id.str.contains("_seed0")]   # model selection uses seed 0 only
order = {"C1": 0, "C2": 1, "C4": 2, "Cm": 3}

# ---- full grid, per forget set: validation J for each (gamma, kappa)
for tag in sorted(df.tag.unique(), key=order.get):
    g = df[(df.tag == tag) & (df.family == "adapt")]
    if len(g):
        print(f"\nJ_val grid for {tag} (rows gamma, cols kappa):")
        print(g.pivot_table(index="gamma", columns="kappa", values="J_val").round(3).to_string())

# ---- best per (tag, family)
best = (df.sort_values(["J_val", "total_time_s"], ascending=[False, True])
          .groupby(["tag", "family"]).head(1))
best = best.sort_values(["tag", "J_val"], key=lambda s: s.map(order) if s.name == "tag" else -s)
cols = ["tag", "family", "gamma", "kappa", "eta", "vol", "repair_epochs_used", "J_val", "MPD",
        "total_time_s", "test_forget_base", "test_forget_impair", "test_forget_final",
        "test_retain_acc_final", "test_retain_f1_final"]
table = best[cols].round(4)
print("\nBest config per family (selected on VALIDATION J, numbers below are TEST):")
print(table.to_string(index=False))
table.to_csv(out / "adaptive_best.csv", index=False)

mapping = {}
for _, r in best.iterrows():
    mapping.setdefault(r["family"], {})[r["tag"]] = r["suffix"]
json.dump(mapping, open(out / "adaptive_best.json", "w"), indent=2)
print("Saved adaptive_best.csv and adaptive_best.json to", out)