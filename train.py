"""
Oil Slick Detection — ResNet-18 Baseline
Trains on random and geographic splits, saves checkpoints and plots.
Run: conda activate oilslick && python train.py
"""

import datetime
import json
import os
import matplotlib
matplotlib.use("Agg")  # headless server — no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.models as models
from collections import Counter
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score, roc_auc_score,
)
from torch.utils.data import Dataset, DataLoader

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = "/mnt/data/home/sf2522/oilslick-detection"
OILSLICK  = os.path.join(BASE_DIR, "data", "data", "OilSlick")
img_dir   = os.path.join(OILSLICK, "images_s1")

RUN_ID    = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR   = os.path.join(BASE_DIR, "results", RUN_ID)
CKPT_DIR  = os.path.join(RUN_DIR, "checkpoints")
PLOT_DIR  = os.path.join(RUN_DIR, "plots")

BATCH_SIZE = 32

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

_log_file = open(os.path.join(RUN_DIR, "train.log"), "a", buffering=1)

def log(msg=""):
    print(msg)
    _log_file.write(msg + "\n")

print(f"Run directory: {RUN_DIR}")

sns.set_style("whitegrid")


def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, name), dpi=120)
    plt.close()
    print(f"  Saved plot: plots/{name}")


# ── 1  Sanity check ──────────────────────────────────────────────────────────
def check_data():
    tif_count = len([f for f in os.listdir(img_dir) if f.endswith(".tif")])
    print(f"img_dir  : {img_dir}")
    print(f"TIF files: {tif_count}  (expected 1363)")
    print(f"metadata : {os.path.exists(os.path.join(OILSLICK, 'metadata.csv'))}")
    print(f"splits   : {os.path.exists(os.path.join(OILSLICK, 'splits'))}")

    files = os.listdir(img_dir)
    tif_files = [f for f in files if f.endswith(".tif")]
    prefixes = Counter(
        f.split("_")[0] + "_" + f.split("_")[1] if f.startswith("ext")
        else f.split("_")[0]
        for f in tif_files
    )
    print("Files by prefix:")
    for prefix, count in sorted(prefixes.items()):
        print(f"  {prefix}: {count}")


# ── 2  Metadata ──────────────────────────────────────────────────────────────
def load_metadata():
    meta_full = pd.read_csv(os.path.join(OILSLICK, "metadata.csv"))
    full_label_map = dict(zip(meta_full["sample_id"], meta_full["label"]))
    print(f"Full label map: {len(full_label_map)} entries")

    meta = meta_full.copy()
    meta = meta[~meta["sample_id"].str.startswith("ext")].reset_index(drop=True)

    available_ids = set(
        f.replace("_s1.tif", "") for f in os.listdir(img_dir) if f.endswith(".tif")
    )
    meta = meta[meta["sample_id"].isin(available_ids)].reset_index(drop=True)
    print(f"Filtered metadata: {len(meta)} rows")
    print(meta["label"].value_counts().to_string())

    return meta, full_label_map


# ── 3  Exploratory plots ─────────────────────────────────────────────────────
def plot_class_distribution(meta):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    meta["label"].value_counts().plot.bar(ax=axes[0])
    axes[0].set_title("Overall Class Distribution")
    axes[0].set_xlabel("Label (0=neg, 1=pos)")
    axes[0].set_ylabel("Count")

    meta["prefix"] = meta["sample_id"].str.extract(r"^(pos|neg|ext_pos|ext_neg)", expand=False)
    meta.groupby(["prefix", "label"]).size().unstack(fill_value=0).plot.bar(ax=axes[1])
    axes[1].set_title("Samples by Prefix & Label")
    savefig("class_distribution.png")


def plot_geo_distribution(meta):
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        meta["center_lon"], meta["center_lat"],
        c=meta["label"], cmap="RdYlGn_r", alpha=0.6, s=15,
        edgecolors="k", linewidths=0.3,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Sample Locations (green=neg, red=pos)")
    plt.colorbar(scatter, ax=ax, label="Label")
    savefig("geo_distribution.png")


def load_chip(sample_id):
    path = os.path.join(img_dir, f"{sample_id}_s1.tif")
    with rasterio.open(path) as src:
        return src.read()  # (2, 224, 224)


def plot_sample_chips(meta):
    n_show = 3
    pos_pool = meta[meta["label"] == 1]["sample_id"]
    neg_pool = meta[meta["label"] == 0]["sample_id"]
    pos_ids = pos_pool.sample(min(n_show, len(pos_pool)), random_state=41).tolist()
    neg_ids = neg_pool.sample(min(n_show, len(neg_pool)), random_state=41).tolist()

    for sample_ids, labels, title, fname in [
        (pos_ids, [1]*len(pos_ids), "Positive Samples (Oil Slick Present)", "chips_positive.png"),
        (neg_ids, [0]*len(neg_ids), "Negative Samples (No Oil Slick)",      "chips_negative.png"),
    ]:
        if not sample_ids:
            continue
        n = len(sample_ids)
        fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
        if n == 1:
            axes = axes[np.newaxis, :]
        fig.suptitle(title, fontsize=14, y=1.01)
        for i, (sid, lab) in enumerate(zip(sample_ids, labels)):
            chip = load_chip(sid)
            for j, band_name in enumerate(["VV", "VH"]):
                ax = axes[i, j]
                band = chip[j]
                valid_px = band[band != 0]
                vmin, vmax = np.percentile(valid_px, [2, 98]) if len(valid_px) > 0 else (band.min(), band.max())
                ax.imshow(band, cmap="gray", vmin=vmin, vmax=vmax)
                ax.set_title(f"{sid}\n{band_name}  label={lab}", fontsize=9)
                ax.axis("off")
        savefig(fname)


def plot_channel_stats(meta):
    print("Computing per-channel statistics on 200 sample chips...")
    rng = np.random.RandomState(0)
    subset = rng.choice(meta["sample_id"].values, size=min(200, len(meta)), replace=False)

    all_vv, all_vh = [], []
    for sid in subset:
        chip = load_chip(sid)
        valid = chip != 0
        all_vv.append(chip[0][valid[0]])
        all_vh.append(chip[1][valid[1]])

    all_vv = np.concatenate(all_vv)
    all_vh = np.concatenate(all_vh)
    print(f"VV — mean: {all_vv.mean():.4f}, std: {all_vv.std():.4f}, "
          f"min: {all_vv.min():.4f}, max: {all_vv.max():.4f}")
    print(f"VH — mean: {all_vh.mean():.4f}, std: {all_vh.std():.4f}, "
          f"min: {all_vh.min():.4f}, max: {all_vh.max():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(all_vv, bins=100, color="steelblue", alpha=0.7)
    axes[0].set_title("VV Backscatter Distribution")
    axes[1].hist(all_vh, bins=100, color="darkorange", alpha=0.7)
    axes[1].set_title("VH Backscatter Distribution")
    savefig("channel_stats.png")


# ── 4  Dataset ───────────────────────────────────────────────────────────────
class OilSlickDataset(Dataset):
    def __init__(self, split_file, img_dir, label_map,
                 vv_mean, vv_std, vh_mean, vh_std, augment=False, noise_std=0.02):
        available = set(
            f.replace("_s1.tif", "") for f in os.listdir(img_dir) if f.endswith(".tif")
        )
        with open(split_file) as f:
            all_ids = [line.strip() for line in f if line.strip()]
        self.sample_ids = [sid for sid in all_ids
                           if sid in available and sid in label_map]
        skipped = len(all_ids) - len(self.sample_ids)
        if skipped > 0:
            print(f"  [{split_file.split('/')[-1]}] Skipped {skipped}, "
                  f"using {len(self.sample_ids)}")
        self.img_dir = img_dir
        self.label_map = label_map
        self.vv_mean, self.vv_std = vv_mean, vv_std
        self.vh_mean, self.vh_std = vh_mean, vh_std
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        path = os.path.join(self.img_dir, f"{sid}_s1.tif")
        with rasterio.open(path) as src:
            chip = src.read().astype(np.float32)
        chip[chip == -163.0] = 0.0
        chip[0] = np.clip(chip[0], -50, 10)
        chip[1] = np.clip(chip[1], -50, 10)
        chip[0] = (chip[0] - self.vv_mean) / (self.vv_std + 1e-8)
        chip[1] = (chip[1] - self.vh_mean) / (self.vh_std + 1e-8)
        chip = torch.from_numpy(chip)
        if self.augment:
            if torch.rand(1).item() > 0.5:
                chip = torch.flip(chip, dims=[2])
            if torch.rand(1).item() > 0.5:
                chip = torch.flip(chip, dims=[1])
            k = torch.randint(0, 4, (1,)).item()
            chip = torch.rot90(chip, k, dims=[1, 2])
            chip = chip + self.noise_std * torch.randn_like(chip)
        label = torch.tensor(self.label_map[sid], dtype=torch.float32)
        return chip, label


# ── 5  Channel stats (Welford) ───────────────────────────────────────────────
def compute_channel_stats(split_file, img_dir):
    available = set(
        f.replace("_s1.tif", "") for f in os.listdir(img_dir) if f.endswith(".tif")
    )
    with open(split_file) as f:
        train_ids = [line.strip() for line in f if line.strip()]
    train_ids = [sid for sid in train_ids if sid in available]
    print(f"Computing stats over {len(train_ids)} training chips...")

    vv_count = vv_mean = vv_M2 = 0.0
    vh_count = vh_mean = vh_M2 = 0.0

    for i, sid in enumerate(train_ids):
        path = os.path.join(img_dir, f"{sid}_s1.tif")
        with rasterio.open(path) as src:
            chip = src.read().astype(np.float32)
        chip[chip == -163.0] = np.nan
        chip[0] = np.clip(chip[0], -50, 10)
        chip[1] = np.clip(chip[1], -50, 10)

        vv_pixels = chip[0][~np.isnan(chip[0])].ravel()
        n = len(vv_pixels)
        if n > 0:
            bm = vv_pixels.mean(); bv = vv_pixels.var()
            d = bm - vv_mean
            vv_mean = (vv_mean * vv_count + bm * n) / (vv_count + n)
            vv_M2  += bv * n + d**2 * vv_count * n / (vv_count + n)
            vv_count += n

        vh_pixels = chip[1][~np.isnan(chip[1])].ravel()
        n = len(vh_pixels)
        if n > 0:
            bm = vh_pixels.mean(); bv = vh_pixels.var()
            d = bm - vh_mean
            vh_mean = (vh_mean * vh_count + bm * n) / (vh_count + n)
            vh_M2  += bv * n + d**2 * vh_count * n / (vh_count + n)
            vh_count += n

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(train_ids)} chips...")

    if vv_count == 0 or vh_count == 0:
        raise RuntimeError("No valid pixels found in training split — check img_dir and split file.")
    print("Done ✓")
    return float(vv_mean), float(np.sqrt(vv_M2 / vv_count)), \
           float(vh_mean), float(np.sqrt(vh_M2 / vh_count))


# ── 6  DataLoaders ───────────────────────────────────────────────────────────
def make_loaders(split_name, vv_mean, vv_std, vh_mean, vh_std, full_label_map,
                 noise_std=0.02, batch_size=32):
    split_dir = os.path.join(OILSLICK, "splits", split_name)
    ds_train = OilSlickDataset(os.path.join(split_dir, "train.txt"), img_dir,
                               full_label_map, vv_mean, vv_std, vh_mean, vh_std,
                               augment=True, noise_std=noise_std)
    ds_val   = OilSlickDataset(os.path.join(split_dir, "val.txt"),   img_dir,
                               full_label_map, vv_mean, vv_std, vh_mean, vh_std)
    ds_test  = OilSlickDataset(os.path.join(split_dir, "test.txt"),  img_dir,
                               full_label_map, vv_mean, vv_std, vh_mean, vh_std)
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    loader_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    loader_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    print(f"[{split_name}] train={len(ds_train)}, val={len(ds_val)}, test={len(ds_test)}")
    return loader_train, loader_val, loader_test


# ── 7  Models ────────────────────────────────────────────────────────────────
def _adapt_conv1(net, pretrained_init):
    old = net.conv1
    new = nn.Conv2d(2, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, padding=old.padding, bias=False)
    if pretrained_init:
        with torch.no_grad():
            w = old.weight.data
            new.weight[:, 0] = w[:, 0] + w[:, 2] * 0.5
            new.weight[:, 1] = w[:, 1] + w[:, 2] * 0.5
    net.conv1 = new


def build_resnet18_2ch(pretrained_init=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained_init else None
    net = models.resnet18(weights=weights)
    _adapt_conv1(net, pretrained_init)
    net.fc = nn.Linear(net.fc.in_features, 1)
    return net


def build_resnet18_v2(pretrained_init=True, dropout=0.4):
    weights = models.ResNet18_Weights.DEFAULT if pretrained_init else None
    net = models.resnet18(weights=weights)
    _adapt_conv1(net, pretrained_init)
    net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 1))
    return net


# ── 8  Evaluate ──────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    all_logits, all_labels, total_loss = [], [], 0.0
    with torch.no_grad():
        for chips, labels in loader:
            chips, labels = chips.to(device), labels.to(device)
            logits = model(chips).squeeze(-1)
            total_loss += criterion(logits, labels).item() * len(labels)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    probs = torch.sigmoid(all_logits).numpy()
    labels_np = all_labels.numpy().astype(int)
    preds = (probs >= 0.5).astype(int)
    return {
        "loss":     total_loss / len(all_labels),
        "accuracy": accuracy_score(labels_np, preds),
        "f1":       f1_score(labels_np, preds, zero_division=0),
        "auroc":    roc_auc_score(labels_np, probs)
                    if len(np.unique(labels_np)) > 1 else 0.0,
        "probs": probs, "labels": labels_np, "preds": preds,
    }


# ── 9  Train ─────────────────────────────────────────────────────────────────
def train_model_v2(model, train_loader, val_loader, device,
                   epochs=60, lr=1e-4, weight_decay=1e-3, patience=20,
                   label_smoothing=0.05, monitor="f1",
                   checkpoint_path=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_auroc": []}
    best_score, best_state, wait = 0.0, None, 0
    start_epoch = 1

    # Resume from checkpoint if one exists
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        best_score  = ckpt["best_score"]
        best_state  = ckpt["model_state"]  # so restore works if no improvement after resume
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from checkpoint: epoch {ckpt['epoch']}, "
              f"best {monitor.upper()}={best_score:.4f}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        for chips, labels in train_loader:
            chips, labels = chips.to(device), labels.to(device)
            labels_smooth = labels * (1 - label_smoothing) + 0.5 * label_smoothing
            optimizer.zero_grad()
            logits = model(chips).squeeze(-1)
            loss = criterion(logits, labels_smooth)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)

        train_loss = running_loss / len(train_loader.dataset)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_auroc"].append(val_metrics["auroc"])

        current_score = val_metrics["f1"] if monitor == "f1" else val_metrics["auroc"]
        lr_now = optimizer.param_groups[0]["lr"]
        log(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  "
            f"val_F1={val_metrics['f1']:.4f}  val_AUROC={val_metrics['auroc']:.4f}  "
            f"lr={lr_now:.2e}"
            + (" ✓" if current_score > best_score else "")
        )

        if current_score > best_score:
            best_score = current_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            if checkpoint_path is not None:
                torch.save({
                    "epoch": epoch,
                    "model_state": best_state,
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_score": best_score,
                }, checkpoint_path)
                log(f"  Checkpoint saved (epoch {epoch}, "
                    f"{monitor.upper()}={best_score:.4f})")
        else:
            wait += 1
            if wait >= patience:
                log(f"Early stopping at epoch {epoch} "
                    f"(best val {monitor.upper()} = {best_score:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    return model, history


# ── 10  Full evaluation ───────────────────────────────────────────────────────
def full_eval(model, loader, device, split_name, plot_suffix=""):
    results = evaluate(model, loader, device)
    print(f"\n{'=' * 55}")
    print(f"  {split_name}")
    print(f"{'=' * 55}")
    print(f"  Accuracy : {results['accuracy']:.4f}")
    print(f"  F1 Score : {results['f1']:.4f}")
    print(f"  AUROC    : {results['auroc']:.4f}")
    print()
    print(classification_report(results["labels"], results["preds"],
                                 target_names=["No Slick", "Oil Slick"]))
    cm = confusion_matrix(results["labels"], results["preds"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No Slick", "Oil Slick"]).plot(
        ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {split_name}")
    savefig(f"confusion_matrix{plot_suffix}.png")
    return results


def plot_training_curves(history, split_name, plot_suffix=""):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title(f"Loss — {split_name}")
    axes[0].legend()
    axes[1].plot(history["val_f1"],    color="green")
    axes[1].set_title("Validation F1")
    axes[2].plot(history["val_auroc"], color="purple")
    axes[2].set_title("Validation AUROC")
    for ax in axes:
        ax.set_xlabel("Epoch")
    savefig(f"training_curves{plot_suffix}.png")


def show_failures(results, split_name, loader, plot_suffix="", n=4):
    probs, labels, preds = results["probs"], results["labels"], results["preds"]
    wrong = np.where(preds != labels)[0]
    if len(wrong) == 0:
        print(f"No misclassifications on {split_name}!")
        return
    confidence = np.abs(probs[wrong] - 0.5)
    top_wrong = wrong[np.argsort(-confidence)[:n]]
    dataset = loader.dataset
    fig, axes = plt.subplots(len(top_wrong), 2, figsize=(7, 3.5 * len(top_wrong)))
    if len(top_wrong) == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f"Most Confident Errors — {split_name}", fontsize=13, y=1.01)
    for i, idx in enumerate(top_wrong):
        sid = dataset.sample_ids[idx]
        chip = load_chip(sid)
        for j, band in enumerate(["VV", "VH"]):
            ax = axes[i, j]
            b = chip[j]
            valid_px = b[b != 0]
            vmin, vmax = np.percentile(valid_px, [2, 98]) if len(valid_px) > 0 else (b.min(), b.max())
            ax.imshow(b, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(
                f"{sid}\n{band} | true={labels[idx]} pred={probs[idx]:.2f}",
                fontsize=9)
            ax.axis("off")
    savefig(f"failures{plot_suffix}.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ── Training config — edit here to change hyperparameters ─────────────────
    CONFIG = {
        "run_id":           RUN_ID,
        "device":           str(device),
        "batch_size":       BATCH_SIZE,
        "split": {
            "model":            "resnet18_v2",
            "pretrained_init":  True,
            "dropout":          0.4,
            "epochs":           80,
            "lr":               5e-5,
            "weight_decay":     5e-4,
            "patience":         20,
            "label_smoothing":  0.05,
            "monitor":          "f1",
            "lr_schedule":      "CosineAnnealingLR",
        },
        "preprocessing": {
            "nodata_sentinel":  -163.0,
            "clip_min":         -50,
            "clip_max":         10,
            "normalization":    "z-score per channel (Welford on train split)",
        },
        "augmentation": {
            "horizontal_flip":  True,
            "vertical_flip":    True,
            "rot90":            True,
            "gaussian_noise_std": 0.02,
        },
    }

    with open(os.path.join(RUN_DIR, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)
    print(f"Config saved to {RUN_DIR}/config.json")

    check_data()
    meta, full_label_map = load_metadata()

    print("\n── Exploratory plots ──")
    plot_class_distribution(meta)
    plot_geo_distribution(meta)
    plot_sample_chips(meta)
    plot_channel_stats(meta)

    # ── Random split ──────────────────────────────────────────────────────────
    print("\n── Random split: channel stats ──")
    random_train_file = os.path.join(OILSLICK, "splits", "random", "train.txt")
    vv_mean, vv_std, vh_mean, vh_std = compute_channel_stats(random_train_file, img_dir)
    print(f"VV: μ={vv_mean:.4f} σ={vv_std:.4f}  |  VH: μ={vh_mean:.4f} σ={vh_std:.4f}")

    # Save channel stats so they can be reused for inference
    with open(os.path.join(RUN_DIR, "channel_stats.json"), "w") as f:
        json.dump({"vv_mean": vv_mean, "vv_std": vv_std,
                   "vh_mean": vh_mean, "vh_std": vh_std}, f, indent=2)

    rand_train, rand_val, rand_test = make_loaders(
        "random", vv_mean, vv_std, vh_mean, vh_std, full_label_map,
        noise_std=CONFIG["augmentation"]["gaussian_noise_std"],
        batch_size=CONFIG["batch_size"])

    print("\n── Random split: training ──")
    sc = CONFIG["split"]
    model_v3 = build_resnet18_v2(pretrained_init=sc["pretrained_init"],
                                  dropout=sc["dropout"]).to(device)
    model_v3, history_v3 = train_model_v2(
        model_v3, rand_train, rand_val, device,
        epochs=sc["epochs"], lr=sc["lr"], weight_decay=sc["weight_decay"],
        patience=sc["patience"], label_smoothing=sc["label_smoothing"],
        monitor=sc["monitor"],
        checkpoint_path=os.path.join(CKPT_DIR, "resnet18_v3_random_best.pt"),
    )
    torch.save(model_v3.state_dict(),
               os.path.join(CKPT_DIR, "resnet18_v3_random_final.pt"))
    plot_training_curves(history_v3, "Random Split", "_random")

    print("\n── Random split: evaluation ──")
    rand_results = full_eval(model_v3, rand_test, device,
                             "Random Split (Test)", "_random")
    show_failures(rand_results, "Random Split", rand_test, "_random")

    # ── Geographic split ──────────────────────────────────────────────────────
    print("\n── Geographic split: channel stats ──")
    geo_train_file = os.path.join(OILSLICK, "splits", "geographic", "train.txt")
    geo_vv_m, geo_vv_s, geo_vh_m, geo_vh_s = compute_channel_stats(
        geo_train_file, img_dir)

    geo_train, geo_val, geo_test = make_loaders(
        "geographic", geo_vv_m, geo_vv_s, geo_vh_m, geo_vh_s, full_label_map,
        noise_std=CONFIG["augmentation"]["gaussian_noise_std"],
        batch_size=CONFIG["batch_size"])

    print("\n── Geographic split: training ──")
    model_geo = build_resnet18_v2(pretrained_init=sc["pretrained_init"],
                                   dropout=sc["dropout"]).to(device)
    model_geo, history_geo = train_model_v2(
        model_geo, geo_train, geo_val, device,
        epochs=sc["epochs"], lr=sc["lr"], weight_decay=sc["weight_decay"],
        patience=sc["patience"], label_smoothing=sc["label_smoothing"],
        monitor=sc["monitor"],
        checkpoint_path=os.path.join(CKPT_DIR, "resnet18_geo_best.pt"),
    )
    torch.save(model_geo.state_dict(),
               os.path.join(CKPT_DIR, "resnet18_geo_final.pt"))
    plot_training_curves(history_geo, "Geographic Split", "_geo")

    print("\n── Geographic split: evaluation ──")
    geo_results = full_eval(model_geo, geo_test, device,
                            "Geographic Split (Mediterranean OOD)", "_geo")
    show_failures(geo_results, "Geographic Split (OOD)", geo_test, "_geo")

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = pd.DataFrame({
        "Metric":                ["Accuracy", "F1 Score", "AUROC"],
        "Random Split":          [f"{rand_results['accuracy']:.4f}",
                                  f"{rand_results['f1']:.4f}",
                                  f"{rand_results['auroc']:.4f}"],
        "Geographic Split (OOD)":[f"{geo_results['accuracy']:.4f}",
                                  f"{geo_results['f1']:.4f}",
                                  f"{geo_results['auroc']:.4f}"],
    })
    print("\n" + "=" * 55)
    print("  ResNet-18 Baseline — Results Summary")
    print("=" * 55)
    print(summary.to_string(index=False))

    # ── Save results to JSON ──────────────────────────────────────────────────
    results_path = os.path.join(RUN_DIR, "results.json")
    results_out = {
        "random_split": {
            "accuracy": round(rand_results["accuracy"], 4),
            "f1":       round(rand_results["f1"],       4),
            "auroc":    round(rand_results["auroc"],     4),
        },
        "geographic_split": {
            "accuracy": round(geo_results["accuracy"], 4),
            "f1":       round(geo_results["f1"],       4),
            "auroc":    round(geo_results["auroc"],     4),
        },
    }
    with open(results_path, "w") as f:
        json.dump(results_out, f, indent=2)
    print(f"\nResults saved to {results_path}")
