"""Mean +/- std over seeds of the verification results (pandas only, no GPU needed)."""
from pathlib import Path

import pandas as pd

from config_loader import load_cfg

cfg = load_cfg()
out = Path(cfg.get("out_dir", "outputs"))
df = pd.read_csv(out / "verification.csv").drop_duplicates(["tag", "method", "seed"], keep="last")
order_t = {"C1": 0, "C2": 1, "C4": 2, "Cm": 3}
order_m = {m: i for i, m in enumerate(["baseline", "retrain", "finetune_only", "control_v2",
                                       "fixed", "vol_sup", "nonoise_sup"])}
cols = {"mia_auc": 3, "mia_tpr1": 3, "probe_bal_acc": 3, "pmass_member": 4, "pmass_nonmember": 4,
        "KL_forget": 2, "KL_retain": 2, "acc_forget": 3, "acc_retain": 3}
rows = []
for (tag, method), g in df.groupby(["tag", "method"]):
    r = {"tag": tag, "method": method, "n": len(g)}
    for c, nd in cols.items():
        m = g[c].mean()
        r[c] = f"{m:.{nd}f}" + (f"±{g[c].std(ddof=1):.{nd}f}" if len(g) > 1 else "")
    rows.append(r)
table = pd.DataFrame(rows).sort_values(["tag", "method"], key=lambda s: s.map(
    order_t if s.name == "tag" else order_m))
table = table.rename(columns={"mia_auc": "MIA_AUC", "mia_tpr1": "TPR@1%FPR", "probe_bal_acc": "probe_acc",
                              "pmass_member": "p_forgot(member)", "pmass_nonmember": "p_forgot(nonmem)"})
for tag, g in table.groupby("tag", sort=False):
    print(f"\n=== {tag} ===")
    print(g.drop(columns="tag").to_string(index=False))
table.to_csv(out / "verification_summary.csv", index=False)
print("\nSaved verification_summary.csv to", out)