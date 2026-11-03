import os
import random

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from dataset import CLASSES


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_model(pretrained=True):
    w = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    m = models.resnet50(weights=w)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    return m


def atomic_save(obj, path):
    """Write to a temp file then rename, so a Colab crash never leaves a corrupt checkpoint."""
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_weights(model, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    return model


@torch.no_grad()
def evaluate(model, dl, device, loss_fn=None, max_batches=None, amp=False):
    model.eval()
    ys, ps, total, n = [], [], 0.0, 0
    for i, (x, y) in enumerate(dl):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=amp and device == "cuda"):
            out = model(x)
        if loss_fn is not None:
            total += loss_fn(out.float(), y).item() * len(y)
        n += len(y)
        ps.append(out.argmax(1).cpu())
        ys.append(y.cpu())
    return torch.cat(ys).numpy(), torch.cat(ps).numpy(), total / max(n, 1)