import yaml
import torch
from torchvision.utils import save_image
from dataset import get_loaders, class_weights, CLASSES, MEAN, STD

if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/config.yaml"))
    train_dl, val_dl, test_dl, train_ds = get_loaders(cfg)

    imgs, labels = next(iter(train_dl))
    print("Batch shape:", imgs.shape)
    print("Labels in batch:", torch.bincount(labels, minlength=7).tolist())
    print("Class weights:", dict(zip(CLASSES, class_weights(train_ds.labels).tolist())))

    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    save_image(imgs[:16] * std + mean, "data/sample_batch.png", nrow=4)
    print("Saved data/sample_batch.png")