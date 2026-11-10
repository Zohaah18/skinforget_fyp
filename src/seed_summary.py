"""Mean +/- std over unlearning seeds, per (forget set, config), from results_adaptive.csv (TEST metrics)."""
from pathlib import Path

import pandas as pd

from config_loader import load_cfg

cfg = load_cfg()
out = Path(cfg.get("out_dir", "outputs"))
df = pd.read_csv(out / "results_adaptive.csv").drop_duplicates("run_id", keep="last")
df = df[~df.run_id.str.contains("smoke")].copy()
df["seed"] = df.run_id.str.extract(r"_seed(\d+)")[0].astype(int)
df["revival"] = df.test_forget_final - df.test_forget_impair      # forgetting lost during repair

fmt = {"J_val": 3, "test_forget_impair": 3, "test_forget_final": 3, "revival": 3,
       "test_retain_acc_final": 3, "test_retain_f1_final": 3, "MPD": 2, "total_time_s": 0}
order = {"C1": 0, "C2": 1, "C4": 2, "Cm": 3}
rows = []
for (tag, suf), g in df.groupby(["tag", "suffix"]):
    r = {"tag": tag, "config": suf, "n": len(g), "seeds": ",".join(map(str, sorted(g.seed)))}
    for c, nd in fmt.items():
        m = g[c].mean()
        r[c] = f"{m:.{nd}f}" + (f"±{g[c].std(ddof=1):.{nd}f}" if len(g) > 1 else "")
    rows.append(r)
table = pd.DataFrame(rows).sort_values(["tag", "config"], key=lambda s: s.map(order) if s.name == "tag" else s)
table = table.rename(columns={"test_forget_impair": "forget@impair", "test_forget_final": "forget@final",
                              "test_retain_acc_final": "retain_acc", "test_retain_f1_final": "retain_F1",
                              "total_time_s": "time_s"})
print(table.to_string(index=False))
table.to_csv(out / "seed_summary.csv", index=False)
print("Saved seed_summary.csv to", out)