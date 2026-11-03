from pathlib import Path
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold
from config_loader import load_cfg
cfg = load_cfg()
root = Path(cfg["data_root"])
out = Path(cfg["splits_dir"])
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(root / "HAM10000_metadata")

paths = {}
for part in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
    for p in (root / part).rglob("*.jpg"):
        paths[p.stem] = p.relative_to(root).as_posix()
df["path"] = df["image_id"].map(paths)
assert df["path"].notna().all(), "Some images were not found, check data_root"

sgkf = StratifiedGroupKFold(n_splits=cfg["test_folds"], shuffle=True,
                            random_state=cfg["seed"])
tv_idx, test_idx = next(sgkf.split(df, df["dx"], groups=df["lesion_id"]))
trainval, test = df.iloc[tv_idx], df.iloc[test_idx]

sgkf2 = StratifiedGroupKFold(n_splits=cfg["val_folds"], shuffle=True,
                             random_state=cfg["seed"])
tr_idx, val_idx = next(sgkf2.split(trainval, trainval["dx"],
                                   groups=trainval["lesion_id"]))
train, val = trainval.iloc[tr_idx], trainval.iloc[val_idx]

assert set(train.lesion_id).isdisjoint(val.lesion_id)
assert set(train.lesion_id).isdisjoint(test.lesion_id)
assert set(val.lesion_id).isdisjoint(test.lesion_id)
assert len(train) + len(val) + len(test) == len(df)

cols = ["image_id", "lesion_id", "dx", "path"]
train[cols].to_csv(out / "train.csv", index=False)
val[cols].to_csv(out / "val.csv", index=False)
test[cols].to_csv(out / "test.csv", index=False)

summary = pd.DataFrame({"train": train.dx.value_counts(),
                        "val": val.dx.value_counts(),
                        "test": test.dx.value_counts()}).fillna(0).astype(int)
summary["total"] = summary.sum(axis=1)
print(summary)
print(summary.sum())
print("Unique lesions:", df.lesion_id.nunique())
summary.to_csv(out / "split_summary.csv")