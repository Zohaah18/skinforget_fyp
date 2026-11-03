from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def get_transforms(train: bool, size: int = 224):
    if train:
        return T.Compose([
            T.Resize((size, size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])


class HAMDataset(Dataset):
    def __init__(self, csv_path, data_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.root = Path(data_root)
        self.transform = transform
        self.labels = self.df["dx"].map(CLASS_TO_IDX).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.root / row["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]


def class_weights(labels):
    counts = torch.bincount(torch.tensor(labels), minlength=len(CLASSES)).float()
    return counts.sum() / (len(CLASSES) * counts)


def balanced_sampler(labels):
    labels_t = torch.tensor(labels)
    counts = torch.bincount(labels_t, minlength=len(CLASSES)).float().clamp(min=1)
    sample_w = (1.0 / counts)[labels_t]
    return WeightedRandomSampler(sample_w, num_samples=len(labels), replacement=True)


def get_loaders(cfg):
    d, root = cfg["splits_dir"], cfg["data_root"]
    sz = cfg["image_size"]
    aug = cfg.get("use_augmentation", False)
    train_ds = HAMDataset(f"{d}/train.csv", root, get_transforms(aug, sz))
    val_ds = HAMDataset(f"{d}/val.csv", root, get_transforms(False, sz))
    test_ds = HAMDataset(f"{d}/test.csv", root, get_transforms(False, sz))

    kw = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
              pin_memory=torch.cuda.is_available())
    if cfg.get("use_balanced_sampler", False):
        train_dl = DataLoader(train_ds, sampler=balanced_sampler(train_ds.labels), **kw)
    else:
        train_dl = DataLoader(train_ds, shuffle=True, **kw)
    val_dl = DataLoader(val_ds, shuffle=False, **kw)
    test_dl = DataLoader(test_ds, shuffle=False, **kw)
    return train_dl, val_dl, test_dl, train_ds