from pathlib import Path
import yaml


def load_cfg(base="configs/config.yaml", local="configs/local.yaml"):
    cfg = yaml.safe_load(open(base))
    if Path(local).exists():
        cfg.update(yaml.safe_load(open(local)) or {})
    return cfg