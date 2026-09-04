"""
Oil Slick Detection — TerraMind (GFM) Fine-tuning
Fine-tunes IBM/ESA TerraMind-1.0 (S1GRD modality) as a linear probe and as a
full fine-tune, on both the random and geographic splits. Mirrors train.py's
structure (same splits, preprocessing steps, and metrics) so the two runs
are directly comparable.

Requires: terratorch>=1.2.5 (pip install "terratorch>=1.2.5"), Python 3.11+.
Weights are pulled automatically from Hugging Face on first run
(ibm-esa-geospatial/TerraMind-1.0-base) — needs outbound network access.

Run: conda activate oilslick && python train_terramind.py
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
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score, roc_auc_score,
)
from torch.utils.data import Dataset, DataLoader

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = "/mnt/data/home/sf2522/oilslick-detection"
OILSLICK  = os.path.join(BASE_DIR, "data", "data", "OilSlick")
img_dir   = os.path.join(OILSLICK, "images_s1")

RUN_ID    = "terramind_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR   = os.path.join(BASE_DIR, "results", RUN_ID)
CKPT_DIR  = os.path.join(RUN_DIR, "checkpoints")
PLOT_DIR  = os.path.join(RUN_DIR, "plots")

MODEL_SIZE = "base"   # tiny | small | base | large
BATCH_SIZE = 32

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

_log_file = open(os.path.join(RUN_DIR, "train.log"), "a", buffering=1)


def log(msg=""):
    print(msg)
    _log_file.write(msg + "\n")


log(f"Run directory: {RUN_DIR}")

sns.set_style("whitegrid")


def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, name), dpi=120)
    plt.close()
    log(f"  Saved plot: plots/{name}")


# ── TerraMind's own S1GRD pretraining statistics ─────────────────────────────
# TerraMind's encoder was pretrained with these fixed VV/VH normalisation
# constants (from the IBM/ESA TerraMesh pretraining corpus). We standardise
# with these — not with Welford stats recomputed per split, as done for the
# ResNet-18 baseline — because a frozen (or lightly fine-tuned) pretrained
# encoder expects inputs on the same distribution it was pretrained on.
TERRAMIND_S1GRD_MEAN = {"VV": -12.599, "VH": -20.293}
TERRAMIND_S1GRD_STD  = {"VV": 5.195,   "VH": 5.890}


# ── 1  Metadata ──────────────────────────────────────────────────────────────
def load_metadata():
    meta_full = pd.read_csv(os.path.join(OILSLICK, "metadata.csv"))
    full_label_map = dict(zip(meta_full["sample_id"], meta_full["label"]))
    log(f"Full label map: {len(full_label_map)} entries")
    return full_label_map


def load_chip(sample_id):
    path = os.path.join(img_dir, f"{sample_id}_s1.tif")
    with rasterio.open(path) as src:
        return src.read()  # (2, 224, 224)


# ── 2  Dataset (same preprocessing steps as train.py, TerraMind norm stats) ──
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
            log(f"  [{split_file.split('/')[-1]}] Skipped {skipped}, "
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


def make_loaders(split_name, full_label_map, noise_std=0.02, batch_size=32):
    split_dir = os.path.join(OILSLICK, "splits", split_name)
    vv_mean, vv_std = TERRAMIND_S1GRD_MEAN["VV"], TERRAMIND_S1GRD_STD["VV"]
    vh_mean, vh_std = TERRAMIND_S1GRD_MEAN["VH"], TERRAMIND_S1GRD_STD["VH"]
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
    log(f"[{split_name}] train={len(ds_train)}, val={len(ds_val)}, test={len(ds_test)}")
    return loader_train, loader_val, loader_test


# ── 3  Model: TerraMind encoder + MLP head ───────────────────────────────────
class TerraMindClassifier(nn.Module):
    def __init__(self, backbone, head, freeze_backbone):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.freeze_backbone = freeze_backbone

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            # Keep a frozen backbone deterministic (no dropout/droppath),
            # regardless of the outer model.train()/eval() calls.
            self.backbone.eval()
        return self

    def forward(self, chips):
        image_dict = {"S1GRD": chips}
        if self.freeze_backbone:
            with torch.no_grad():
                out = self.backbone(image_dict)
        else:
            out = self.backbone(image_dict)
        feats = out[-1] if isinstance(out, (list, tuple)) else out  # (B, N, D)
        pooled = feats.mean(dim=1)                                  # (B, D)
        return self.head(pooled).squeeze(-1)


def _infer_embed_dim(backbone):
    backbone.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 2, 224, 224)
        out = backbone({"S1GRD": dummy})
        feats = out[-1] if isinstance(out, (list, tuple)) else out
    return feats.shape[-1]


def build_terramind_classifier(model_size="base", dropout=0.4, freeze_backbone=True):
    import terratorch
    from terratorch.registry import BACKBONE_REGISTRY

    backbone = BACKBONE_REGISTRY.build(
        f"terramind_v1_{model_size}", modalities=["S1GRD"], pretrained=True,
    )
    embed_dim = _infer_embed_dim(backbone)
    log(f"  TerraMind-{model_size} embed_dim={embed_dim}")

    if freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False

    head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, 1))
    return TerraMindClassifier(backbone, head, freeze_backbone)


# ── 4  Evaluate ──────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    all_logits, all_labels, total_loss = [], [], 0.0
    with torch.no_grad():
        for chips, labels in loader:
            chips, labels = chips.to(device), labels.to(device)
            logits = model(chips)
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


# ── 5  Train ─────────────────────────────────────────────────────────────────
def train_terramind_model(model, train_loader, val_loader, device,
                          param_groups, epochs, weight_decay, patience,
                          label_smoothing=0.05, monitor="f1",
                          checkpoint_path=None):
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_auroc": []}
    best_score, best_state, wait = 0.0, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for chips, labels in train_loader:
            chips, labels = chips.to(device), labels.to(device)
            labels_smooth = labels * (1 - label_smoothing) + 0.5 * label_smoothing
            optimizer.zero_grad()
            logits = model(chips)
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
                torch.save({"epoch": epoch, "model_state": best_state,
                            "best_score": best_score}, checkpoint_path)
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


# ── 6  Full evaluation & plotting ────────────────────────────────────────────
def full_eval(model, loader, device, split_name, plot_suffix=""):
    results = evaluate(model, loader, device)
    log(f"\n{'=' * 55}")
    log(f"  {split_name}")
    log(f"{'=' * 55}")
    log(f"  Accuracy : {results['accuracy']:.4f}")
    log(f"  F1 Score : {results['f1']:.4f}")
    log(f"  AUROC    : {results['auroc']:.4f}")
    log("")
    log(classification_report(results["labels"], results["preds"],
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
        log(f"No misclassifications on {split_name}!")
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
    log(f"Using device: {device}\n")

    # ── Config ────────────────────────────────────────────────────────────────
    MODES = {
        "linear_probe": {
            "freeze_backbone": True,
            "epochs": 50,
            "patience": 15,
            "head_lr": 1e-3,
            "backbone_lr": None,       # frozen, no backbone LR
            "weight_decay": 5e-4,
        },
        "full_finetune": {
            "freeze_backbone": False,
            "epochs": 40,
            "patience": 15,
            "head_lr": 1e-4,
            "backbone_lr": 2e-5,       # small LR — pretrained encoder needs only gentle adaptation
            "weight_decay": 5e-4,
        },
    }
    DROPOUT = 0.4
    LABEL_SMOOTHING = 0.05
    NOISE_STD = 0.02
    MONITOR = "f1"

    CONFIG = {
        "run_id": RUN_ID,
        "device": str(device),
        "batch_size": BATCH_SIZE,
        "model_size": MODEL_SIZE,
        "backbone": f"terramind_v1_{MODEL_SIZE}",
        "modalities": ["S1GRD"],
        "dropout": DROPOUT,
        "label_smoothing": LABEL_SMOOTHING,
        "monitor": MONITOR,
        "modes": MODES,
        "preprocessing": {
            "nodata_sentinel": -163.0,
            "clip_min": -50,
            "clip_max": 10,
            "normalization": "z-score per channel using TerraMind S1GRD pretraining stats (fixed, not recomputed per split)",
            "terramind_s1grd_mean": TERRAMIND_S1GRD_MEAN,
            "terramind_s1grd_std": TERRAMIND_S1GRD_STD,
        },
        "augmentation": {
            "horizontal_flip": True, "vertical_flip": True, "rot90": True,
            "gaussian_noise_std": NOISE_STD,
        },
    }
    with open(os.path.join(RUN_DIR, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)
    log(f"Config saved to {RUN_DIR}/config.json")

    full_label_map = load_metadata()

    all_results = {}
    for split_name in ["random", "geographic"]:
        log(f"\n{'#' * 65}\n#  Split: {split_name}\n{'#' * 65}")
        train_loader, val_loader, test_loader = make_loaders(
            split_name, full_label_map, noise_std=NOISE_STD, batch_size=BATCH_SIZE)

        split_results = {}
        for mode_name, mc in MODES.items():
            log(f"\n── {split_name} / {mode_name} ──")
            model = build_terramind_classifier(
                model_size=MODEL_SIZE, dropout=DROPOUT,
                freeze_backbone=mc["freeze_backbone"]).to(device)

            if mc["freeze_backbone"]:
                param_groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                                  "lr": mc["head_lr"]}]
            else:
                param_groups = [
                    {"params": model.backbone.parameters(), "lr": mc["backbone_lr"]},
                    {"params": model.head.parameters(),     "lr": mc["head_lr"]},
                ]

            ckpt_path = os.path.join(CKPT_DIR, f"terramind_{MODEL_SIZE}_{mode_name}_{split_name}_best.pt")
            model, history = train_terramind_model(
                model, train_loader, val_loader, device,
                param_groups=param_groups, epochs=mc["epochs"],
                weight_decay=mc["weight_decay"], patience=mc["patience"],
                label_smoothing=LABEL_SMOOTHING, monitor=MONITOR,
                checkpoint_path=ckpt_path,
            )
            plot_suffix = f"_terramind_{mode_name}_{split_name}"
            plot_training_curves(history, f"{split_name} / {mode_name}", plot_suffix)

            results = full_eval(model, test_loader, device,
                                f"TerraMind-{MODEL_SIZE} {mode_name} — {split_name} (Test)",
                                plot_suffix)
            show_failures(results, f"{split_name}/{mode_name}", test_loader, plot_suffix)

            split_results[mode_name] = {
                "accuracy": round(results["accuracy"], 4),
                "f1":       round(results["f1"], 4),
                "auroc":    round(results["auroc"], 4),
            }
        all_results[f"{split_name}_split"] = split_results

    # ── Summary ───────────────────────────────────────────────────────────────
    rows = []
    for split_key, modes in all_results.items():
        for mode_name, m in modes.items():
            rows.append({"Split": split_key, "Mode": mode_name,
                         "Accuracy": m["accuracy"], "F1": m["f1"], "AUROC": m["auroc"]})
    summary = pd.DataFrame(rows)
    log("\n" + "=" * 65)
    log(f"  TerraMind-{MODEL_SIZE} — Results Summary")
    log("=" * 65)
    log(summary.to_string(index=False))

    results_path = os.path.join(RUN_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump({"model_size": MODEL_SIZE, **all_results}, f, indent=2)
    log(f"\nResults saved to {results_path}")
