"""
run_terramind.py
Trains and evaluates TerraMind on the oil slick dataset.

Strategy: frozen-backbone linear probing via feature pre-extraction.
  1. Backbone runs once on CPU to extract (N, 768) features for every split.
  2. A tiny Linear(768→1) head trains on those cached tensors — no backbone in
     the training loop, so GPU memory usage is negligible.

Loads pre-saved ResNet-18 metrics from results/2026-06-13_15-36-23/results.json
for the comparison table — no ResNet retraining needed.
"""

import os, json
from datetime import datetime
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = "/mnt/data/home/sf2522/oilslick-detection"
OILSLICK   = os.path.join(BASE_DIR, "data", "data", "OilSlick")
img_dir    = os.path.join(OILSLICK, "images_s1")
RESNET_DIR = os.path.join(BASE_DIR, "results", "2026-06-13_15-36-23")

RUN_DIR    = os.path.join(BASE_DIR, "results", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
CKPT_DIR   = os.path.join(RUN_DIR, "checkpoints")
PLOTS_DIR  = os.path.join(RUN_DIR, "plots")
FEAT_DIR   = os.path.join(RUN_DIR, "features")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(FEAT_DIR, exist_ok=True)
print(f"Saving results to: {RUN_DIR}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Label map ─────────────────────────────────────────────────────────────────
import pandas as pd
meta_full = pd.read_csv(os.path.join(OILSLICK, "metadata.csv"))
full_label_map = dict(zip(meta_full["sample_id"], meta_full["label"]))

# ── Load pre-saved ResNet-18 results ──────────────────────────────────────────
with open(os.path.join(RESNET_DIR, "results.json")) as f:
    resnet_results = json.load(f)
print(f"Loaded ResNet-18 results from {RESNET_DIR}/results.json")
print(f"  Random     — F1={resnet_results['random_split']['f1']:.4f}  "
      f"AUROC={resnet_results['random_split']['auroc']:.4f}")
print(f"  Geographic — F1={resnet_results['geographic_split']['f1']:.4f}  "
      f"AUROC={resnet_results['geographic_split']['auroc']:.4f}")


# ── Utilities ─────────────────────────────────────────────────────────────────
def evaluate_head(model, feats, labels, device):
    model.eval()
    feats, labels = feats.to(device), labels.to(device)
    with torch.no_grad():
        logits = model(feats).squeeze(-1)
    probs    = torch.sigmoid(logits).cpu().numpy()
    labels_np = labels.cpu().numpy().astype(int)
    preds     = (probs >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(labels_np, preds),
        "f1":       f1_score(labels_np, preds, zero_division=0),
        "auroc":    roc_auc_score(labels_np, probs) if len(np.unique(labels_np)) > 1 else 0.0,
        "probs":    probs,
        "labels":   labels_np,
        "preds":    preds,
    }


def train_head(head, tr_feats, tr_labels, va_feats, va_labels, device,
               epochs=200, lr=1e-3, patience=30, label_smoothing=0.05,
               checkpoint_path=None):
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()
    history   = {"train_loss": [], "val_f1": [], "val_auroc": []}
    best_f1, best_state, wait = 0.0, None, 0

    tr_feats, tr_labels = tr_feats.to(device), tr_labels.to(device)
    va_feats, va_labels = va_feats.to(device), va_labels.to(device)

    dataset = TensorDataset(tr_feats, tr_labels)
    loader  = DataLoader(dataset, batch_size=256, shuffle=True)

    epoch_bar = tqdm(range(1, epochs + 1), desc="Training head", unit="epoch",
                     dynamic_ncols=True)
    for epoch in epoch_bar:
        head.train()
        running_loss = 0.0
        for xb, yb in loader:
            smooth = yb * (1 - label_smoothing) + 0.5 * label_smoothing
            optimizer.zero_grad()
            loss = criterion(head(xb).squeeze(-1), smooth)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(yb)
        train_loss = running_loss / len(tr_labels)
        scheduler.step()

        val_m = evaluate_head(head, va_feats, va_labels, device)
        history["train_loss"].append(train_loss)
        history["val_f1"].append(val_m["f1"])
        history["val_auroc"].append(val_m["auroc"])

        improved = val_m["f1"] > best_f1
        epoch_bar.set_postfix(
            train=f"{train_loss:.4f}",
            val_F1=f"{val_m['f1']:.4f}",
            val_AUROC=f"{val_m['auroc']:.4f}",
            best=f"{max(best_f1, val_m['f1']):.4f}",
            wait=0 if improved else wait + 1,
        )
        if improved:
            best_f1 = val_m["f1"]
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            wait = 0
            if checkpoint_path:
                torch.save({"epoch": epoch, "model_state": best_state,
                            "best_f1": best_f1}, checkpoint_path)
        else:
            wait += 1
            if wait >= patience:
                tqdm.write(f"Early stopping at epoch {epoch} (best F1={best_f1:.4f})")
                break

    if best_state:
        head.load_state_dict(best_state)
        head.to(device)
    return head, history


def plot_and_save_curves(history, title, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["train_loss"]); axes[0].set_title(f"{title} — Loss")
    axes[1].plot(history["val_f1"],    color="green");  axes[1].set_title(f"{title} — Val F1")
    axes[2].plot(history["val_auroc"], color="purple"); axes[2].set_title(f"{title} — Val AUROC")
    for ax in axes: ax.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved → {path}")


def print_and_save_eval(results, title, cm_path):
    print(f"\n{'='*55}\n  {title}\n{'='*55}")
    print(f"  Accuracy : {results['accuracy']:.4f}")
    print(f"  F1 Score : {results['f1']:.4f}")
    print(f"  AUROC    : {results['auroc']:.4f}")
    print(classification_report(results["labels"], results["preds"],
                                target_names=["No Slick", "Oil Slick"]))
    cm = confusion_matrix(results["labels"], results["preds"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No Slick", "Oil Slick"]).plot(ax=ax, cmap="Blues")
    ax.set_title(title); plt.tight_layout()
    plt.savefig(cm_path, dpi=150); plt.close()
    print(f"Saved → {cm_path}")


# ── Raw image dataset (no augmentation — used only for feature extraction) ────
# TerraMind v1 pretraining stats for untok_sen1grd@224 (VV, VH, in dB)
_TM_MEAN = torch.tensor([-12.599, -20.293]).view(2, 1, 1)
_TM_STD  = torch.tensor([ 5.195,   5.890]).view(2, 1, 1)

class OilSlickDatasetRaw(Dataset):
    def __init__(self, split_file, img_dir, label_map):
        available = set(f.replace("_s1.tif", "") for f in os.listdir(img_dir) if f.endswith(".tif"))
        with open(split_file) as f:
            ids = [l.strip() for l in f if l.strip()]
        self.sample_ids = [s for s in ids if s in available and s in label_map]
        self.img_dir = img_dir
        self.label_map = label_map
        print(f"  [{os.path.basename(split_file)}] {len(self.sample_ids)} samples")

    def __len__(self): return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        with rasterio.open(os.path.join(self.img_dir, f"{sid}_s1.tif")) as src:
            chip = src.read().astype(np.float32)
        chip[chip == -163.0] = 0.0
        chip[0] = np.clip(chip[0], -50, 10)
        chip[1] = np.clip(chip[1], -50, 10)
        chip = torch.from_numpy(chip)
        chip = (chip - _TM_MEAN) / _TM_STD   # z-score to match TerraMind pretraining
        return chip, torch.tensor(self.label_map[sid], dtype=torch.float32)


# ── TerraMind backbone (CPU — kept out of training loop entirely) ─────────────
from terratorch.registry import BACKBONE_REGISTRY

def build_backbone():
    print("Loading TerraMind backbone (CPU)...")
    backbone = BACKBONE_REGISTRY.build(
        "terramind_v1_base",
        pretrained=True,
        modalities=[{"untok_sen1grd@224": 2}],
    )
    backbone.eval()
    return backbone


def extract_features(backbone, split_file, tag):
    """Extract (N, 768) features for one split; saves/loads cache from disk."""
    feat_path  = os.path.join(FEAT_DIR, f"{tag}_feats.pt")
    label_path = os.path.join(FEAT_DIR, f"{tag}_labels.pt")
    if os.path.exists(feat_path):
        print(f"  Loading cached features: {feat_path}")
        return torch.load(feat_path), torch.load(label_path)

    dataset = OilSlickDatasetRaw(split_file, img_dir, full_label_map)
    loader  = DataLoader(dataset, batch_size=16, shuffle=False,
                         num_workers=4, pin_memory=False)

    all_feats, all_labels = [], []
    with torch.no_grad():
        for chips, labels in tqdm(loader, desc=f"  Extracting {tag}", unit="batch"):
            out   = backbone(chips)          # CPU forward — no CUDA needed
            feats = out[-1].mean(dim=1)      # (B, 768)
            all_feats.append(feats)
            all_labels.append(labels)

    feats_t  = torch.cat(all_feats)
    labels_t = torch.cat(all_labels)
    torch.save(feats_t,  feat_path)
    torch.save(labels_t, label_path)
    print(f"  Saved features → {feat_path}  shape={tuple(feats_t.shape)}")
    return feats_t, labels_t


# ── Linear head ───────────────────────────────────────────────────────────────
def make_head(dropout=0.4):
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(768, 1)).to(device)


# ═════════════════════════════════════════════════════════════════════════════
# Feature extraction (backbone runs once on CPU, then is discarded)
# ═════════════════════════════════════════════════════════════════════════════
backbone = build_backbone()   # CPU

splits_dir = os.path.join(OILSLICK, "splits")

print("\nExtracting features — random split")
rand_tr_f, rand_tr_l = extract_features(backbone, f"{splits_dir}/random/train.txt", "rand_train")
rand_va_f, rand_va_l = extract_features(backbone, f"{splits_dir}/random/val.txt",   "rand_val")
rand_te_f, rand_te_l = extract_features(backbone, f"{splits_dir}/random/test.txt",  "rand_test")

print("\nExtracting features — geographic split")
geo_tr_f, geo_tr_l = extract_features(backbone, f"{splits_dir}/geographic/train.txt", "geo_train")
geo_va_f, geo_va_l = extract_features(backbone, f"{splits_dir}/geographic/val.txt",   "geo_val")
geo_te_f, geo_te_l = extract_features(backbone, f"{splits_dir}/geographic/test.txt",  "geo_test")

# Free backbone — not needed for the rest of training
del backbone
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ═════════════════════════════════════════════════════════════════════════════
# 1. Train head — random split
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Training head — random split")
print("="*60)

head_rand = make_head()
head_rand, history_rand = train_head(
    head_rand, rand_tr_f, rand_tr_l, rand_va_f, rand_va_l, device,
    epochs=200, lr=1e-3, patience=30, label_smoothing=0.05,
    checkpoint_path=os.path.join(CKPT_DIR, "terramind_random_best.pt"),
)
rand_results = evaluate_head(head_rand, rand_te_f, rand_te_l, device)
print_and_save_eval(rand_results, "TerraMind — Random Split",
                    os.path.join(PLOTS_DIR, "confusion_matrix_random.png"))
plot_and_save_curves(history_rand, "TerraMind Random",
                     os.path.join(PLOTS_DIR, "training_curves_random.png"))

# ═════════════════════════════════════════════════════════════════════════════
# 2. Train head — geographic split
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Training head — geographic split")
print("="*60)

head_geo = make_head()
head_geo, history_geo = train_head(
    head_geo, geo_tr_f, geo_tr_l, geo_va_f, geo_va_l, device,
    epochs=200, lr=1e-3, patience=30, label_smoothing=0.05,
    checkpoint_path=os.path.join(CKPT_DIR, "terramind_geographic_best.pt"),
)
geo_results = evaluate_head(head_geo, geo_te_f, geo_te_l, device)
print_and_save_eval(geo_results, "TerraMind — Geographic Split (OOD)",
                    os.path.join(PLOTS_DIR, "confusion_matrix_geo.png"))
plot_and_save_curves(history_geo, "TerraMind Geographic",
                     os.path.join(PLOTS_DIR, "training_curves_geo.png"))

# ═════════════════════════════════════════════════════════════════════════════
# 3. Save results
# ═════════════════════════════════════════════════════════════════════════════
def _serialisable(r):
    return {k: v.tolist() if hasattr(v, "tolist") else v
            for k, v in r.items() if k not in ("probs", "labels", "preds")}

results_out = {
    "random_split":     _serialisable(rand_results),
    "geographic_split": _serialisable(geo_results),
}
with open(os.path.join(RUN_DIR, "results.json"), "w") as f:
    json.dump(results_out, f, indent=2)

with open(os.path.join(RUN_DIR, "histories.json"), "w") as f:
    json.dump({"random": history_rand, "geographic": history_geo}, f, indent=2)

print(f"\nResults saved → {RUN_DIR}/results.json")

# ═════════════════════════════════════════════════════════════════════════════
# 4. Comparison table
# ═════════════════════════════════════════════════════════════════════════════
rn = resnet_results
print("\n" + "="*70)
print(f"{'Model':<28} {'Split':<15} {'Accuracy':>9} {'F1':>7} {'AUROC':>7}")
print("-"*70)
print(f"  {'ResNet-18':<26} {'Random':<15} {rn['random_split']['accuracy']:>9.4f} {rn['random_split']['f1']:>7.4f} {rn['random_split']['auroc']:>7.4f}")
print(f"  {'ResNet-18':<26} {'Geographic':<15} {rn['geographic_split']['accuracy']:>9.4f} {rn['geographic_split']['f1']:>7.4f} {rn['geographic_split']['auroc']:>7.4f}")
print(f"  {'TerraMind (linear probe)':<26} {'Random':<15} {rand_results['accuracy']:>9.4f} {rand_results['f1']:>7.4f} {rand_results['auroc']:>7.4f}")
print(f"  {'TerraMind (linear probe)':<26} {'Geographic':<15} {geo_results['accuracy']:>9.4f} {geo_results['f1']:>7.4f} {geo_results['auroc']:>7.4f}")
print("="*70)
print(f"\nAll outputs in: {RUN_DIR}/")
