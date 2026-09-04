# Oil Slick Detection — Baseline CNN Pipeline

Binary classification of **Sentinel-1 SAR satellite imagery** to detect the presence or absence of marine oil slicks. This project builds a ResNet-18 baseline adapted for 2-channel (VV/VH polarisation) radar input and evaluates it under two complementary split strategies: a standard random split and a geographically disjoint out-of-distribution split.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset](#2-dataset)
3. [Data Exploration](#3-data-exploration)
4. [Preprocessing](#4-preprocessing)
5. [Channel Statistics — Welford's Algorithm](#5-channel-statistics--welfords-algorithm)
6. [Dataset Class and DataLoaders](#6-dataset-class-and-dataloaders)
7. [Model Architecture](#7-model-architecture)
8. [Training Loop](#8-training-loop)
9. [Evaluation](#9-evaluation)
10. [Results](#10-results)
11. [Failure Case Analysis](#11-failure-case-analysis)
12. [Standalone Training Script](#12-standalone-training-script)
13. [TerraMind (GFM) Fine-tuning](#13-terramind-gfm-fine-tuning)
14. [Project Structure](#14-project-structure)

---

## 1. Problem Statement

Marine oil slicks are a serious environmental hazard. Detecting them from satellite imagery enables rapid incident response. Sentinel-1 is a radar satellite (SAR — Synthetic Aperture Radar) that operates in all weather and lighting conditions, making it well-suited for operational maritime monitoring.

**Task:** Given a 224×224 pixel GeoTIFF chip with two radar polarisation channels (VV and VH), predict whether an oil slick is present (`label=1`) or absent (`label=0`).

**Why SAR?** Unlike optical sensors, SAR penetrates cloud cover and works at night. Oil slicks dampen ocean surface roughness, producing a distinctive low-backscatter signature visible in both VV and VH channels.

**Key challenge:** A model trained on one geographic region must generalise to unseen ocean areas and environmental conditions — tested via the geographic split.

---

## 2. Dataset

**Source:** WaterBench OilSlick dataset ([`metadata.csv`](data/data/OilSlick/metadata.csv))

The dataset contains 1363 annotated samples across four categories:
- `pos_*` — confirmed oil slick present (486 on disk)
- `neg_*` — confirmed no oil slick (484 on disk)
- `ext_pos_*` / `ext_neg_*` — extended set (11 on disk; excluded from training)

Each sample is a **2-band GeoTIFF** at 224×224 pixels:
- **Band 1: VV** — vertically transmitted, vertically received polarisation
- **Band 2: VH** — vertically transmitted, horizontally received polarisation

```
Cell 2 (notebook):
    BASE_DIR  = "/mnt/data/home/sf2522/oilslick-detection"
    OILSLICK  = os.path.join(BASE_DIR, "data", "data", "OilSlick")
    img_dir   = os.path.join(OILSLICK, "images_s1")

    tif_count = len([f for f in os.listdir(img_dir) if f.endswith(".tif")])
    # → TIF files: 981  (expected 1363)
```

**Note:** Only 981 of 1363 annotated samples are physically on disk — the remainder were not downloaded. The pipeline filters automatically to only use samples present on disk.

**Split files** are pre-generated text files listing sample IDs:

```
data/data/OilSlick/splits/
  random/        train.txt (900 IDs) / val.txt / test.txt
  geographic/    train.txt / val.txt / test.txt
```

After filtering to disk-available samples, the effective random split sizes are:
- **Train:** 655 | **Val:** 97 | **Test:** 224

---

## 3. Data Exploration

> Before building anything, we need to understand what the data actually looks like. This section answers three basic questions: are the classes balanced, where in the world are the samples from, and what do the raw pixel values actually look like? The answers directly shape every decision made in the preprocessing step.

### 3.1 Label Distribution (Cell `a0hnqejQI53d`)

After filtering to disk-available `pos_*` and `neg_*` samples:
- **Positive (oil slick):** 486 samples
- **Negative (no oil slick):** 484 samples

The dataset is effectively **balanced**, so no class-weighting is needed.

### 3.2 Geographic Distribution (Cell `8Gbm2Vw63lHj`)

Samples are drawn from ocean regions worldwide. The geographic split specifically holds out the **Mediterranean Sea** as an out-of-distribution test region, reflecting the real-world scenario where a model trained on one region is deployed elsewhere.

### 3.3 Visualising SAR Chips (Cell `HxsFSjp33lHk`)

Chips are loaded with `rasterio` and displayed using a 2nd–98th percentile stretch to account for the heavy-tailed dB distribution:

```python
def load_chip(sample_id):
    path = os.path.join(img_dir, f"{sample_id}_s1.tif")
    with rasterio.open(path) as src:
        data = src.read()   # shape: (2, 224, 224)
    return data
```

Visual inspection shows oil slicks appear as **dark, smooth patches** with low backscatter against a rougher ocean background. The effect is clearer in VV than VH.

### 3.4 Raw Pixel Value Distribution (Cell `DF9F8FDN3lHk`)

Examining 200 randomly sampled chips reveals:
- **Mean VV: −28.8 dB**, Std: 42.6 — extremely high std driven by outliers
- **Mean VH: −36.8 dB**, Std: 39.1 — similarly dominated by outliers
- **Minimum: −163** — a sentinel nodata value (land mask / missing data)
- **Maximum: ~1997** — extreme bright targets (ships, infrastructure)

These outliers would dominate z-score normalisation and squash the meaningful ocean signal, motivating the clipping step described next.

---

## 4. Preprocessing

> Raw SAR values straight off the satellite are not suitable for a neural network. They contain missing data placeholders, extreme outliers from ships and coastlines, and wildly different value ranges between chips. This section describes how each chip is cleaned and standardised into a form the model can learn from — every step here is motivated by the data exploration findings above.

All preprocessing is applied inside `OilSlickDataset.__getitem__` in a deterministic, per-sample order:

```python
# Step 1 — Mask nodata sentinel (Cell wvdxlDVr3lHl)
chip[chip == -163.0] = 0.0

# Step 2 — Clip to physically meaningful dB range
chip[0] = np.clip(chip[0], -50, 10)   # VV channel
chip[1] = np.clip(chip[1], -50, 10)   # VH channel

# Step 3 — Z-score normalise per channel
chip[0] = (chip[0] - self.vv_mean) / (self.vv_std + 1e-8)
chip[1] = (chip[1] - self.vh_mean) / (self.vh_std + 1e-8)
```

**Why −50 to +10 dB?**
- −50 dB is below the ocean noise floor; anything lower is noise or the nodata edge
- +10 dB is well above the expected ocean return and excludes bright ship/land artefacts
- This range captures the full dynamic range of oil slick vs clean ocean backscatter

**Why nodata = 0 after masking?**
After clipping, valid ocean pixels span [−50, 10] dB. Setting masked pixels to 0.0 before normalisation means they will have a non-zero z-score, but they are spatially confined to borders and do not affect the loss. An alternative would be to use the nodata mask as a loss weight; for this baseline, 0-fill is sufficient.

### Augmentation (training only)

Applied after normalisation, to prevent the network from learning position-specific artefacts:

```python
if self.augment:
    if torch.rand(1).item() > 0.5:
        chip = torch.flip(chip, dims=[2])       # horizontal flip
    if torch.rand(1).item() > 0.5:
        chip = torch.flip(chip, dims=[1])       # vertical flip
    k = torch.randint(0, 4, (1,)).item()
    chip = torch.rot90(chip, k, dims=[1, 2])   # random 90° rotation
    chip = chip + 0.02 * torch.randn_like(chip) # Gaussian noise σ=0.02
```

SAR chips have no preferred orientation (the satellite can overpass from any direction), so flips and 90° rotations are label-preserving. Gaussian noise (σ=0.02 in `train.py`) simulates speckle variation.

---

## 5. Channel Statistics — Welford's Algorithm

> Z-score normalisation requires knowing the mean and standard deviation of the training data — but we can't load all 655 training chips into memory at once (each chip is 224×224×2 float32, and reading them all would require ~1 GB). This section describes how we compute exact statistics in a single pass through the data without ever holding more than one chip in memory at a time.

Normalisation statistics must be computed from the **training split only** to prevent information leakage from val/test. Loading all training pixels into memory would require ~1 GB, so instead we use **Welford's online algorithm** extended for batch-level merging (Cell `PQt0DBKSNUMU`):

```python
def compute_channel_stats(split_file, img_dir):
    # For each training chip:
    #   1. Load and clean (mask nodata, clip to [-50, 10])
    #   2. Extract valid pixels as a flat array
    #   3. Merge this batch's mean and variance into the running total
    #      using the parallel Welford formula:

    delta   = batch_mean - running_mean
    running_mean = (running_mean * n_so_far + batch_mean * n_batch) / (n_so_far + n_batch)
    M2      += batch_var * n_batch + delta**2 * n_so_far * n_batch / (n_so_far + n_batch)
    n_so_far += n_batch

    # Final std:
    std = sqrt(M2 / n_total)
```

This is numerically stable and processes one chip at a time, using O(1) memory regardless of dataset size.

**Computed statistics from the random training split (655 chips):**
```
VV: μ = −20.6575 dB,  σ = 13.7535 dB
VH: μ = −25.7585 dB,  σ = 15.9263 dB
```

These are recomputed from scratch for the geographic split's training set, ensuring normalisation is always split-specific.

---

## 6. Dataset Class and DataLoaders

> PyTorch needs a standardised interface to feed data into the model during training. The `OilSlickDataset` class wraps all the preprocessing and file-reading logic, and `make_loaders` assembles it into batched iterators for train, validation, and test sets. This is the glue between the raw files on disk and the model.

### OilSlickDataset (Cell `wvdxlDVr3lHl`)

```python
class OilSlickDataset(Dataset):
    def __init__(self, split_file, img_dir, label_map,
                 vv_mean, vv_std, vh_mean, vh_std, augment=False):
        # Loads only IDs that are:
        #   (a) listed in the split text file
        #   (b) present on disk as {id}_s1.tif
        #   (c) have a label entry in label_map (from metadata.csv)
```

The label map is built from the **full** `metadata.csv` (all 1363 entries), not just the filtered subset, so IDs from any split file can be looked up regardless of filtering order.

### make_loaders (Cell `19MtBKDU3lHl`)

```python
def make_loaders(split_name, vv_mean, vv_std, vh_mean, vh_std):
    # Creates train (augment=True), val, test (augment=False) DataLoaders
    # batch_size=32, num_workers=0 (rasterio is not fork-safe with GDAL)
    # pin_memory=True for faster GPU transfer
```

**`num_workers=0`:** `rasterio` uses GDAL under the hood, which is not fork-safe. Using multiple workers can cause silent hangs or corrupted reads. Zero workers means data loading happens in the main process.

---

## 7. Model Architecture

> The model is the part that actually learns to distinguish oil slicks from clean ocean. We use ResNet-18, a well-established image classification network, but it needs two modifications before it can work with SAR data: its input layer must be changed from 3 channels (RGB) to 2 channels (VV/VH), and its output layer must be changed from 1000 class scores (ImageNet) to a single oil-slick probability. This section explains how both changes are made without throwing away the pretrained knowledge the network already has.

### 7.1 Why ResNet-18?

ResNet-18 was chosen as the baseline for three reasons:
1. **Mature ImageNet pretraining** — feature extractors generalise well even to non-natural images
2. **Small enough to train fast** — 11.2M parameters, converges in under an hour on a single GPU
3. **Established benchmark** — widely used in remote sensing baselines, making results comparable

### 7.2 Adapting conv1 for 2-Channel Input (Cell `-iqMyPRw3lHm`)

ImageNet-pretrained ResNet-18 expects 3-channel (RGB) input. SAR images have 2 channels (VV, VH). A naive random initialisation of a new conv1 would discard all pretrained knowledge; instead, we **blend** the RGB weights:

```python
def _adapt_conv1(net, pretrained_init):
    old = net.conv1                         # (64, 3, 7, 7) weight tensor
    new = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

    if pretrained_init:
        w = old.weight.data
        new.weight[:, 0] = w[:, 0] + w[:, 2] * 0.5   # VV ← R + 0.5·B
        new.weight[:, 1] = w[:, 1] + w[:, 2] * 0.5   # VH ← G + 0.5·B

    net.conv1 = new
```

**Rationale for the blending formula:**
- VV (co-polarisation, stronger return) is conceptually similar to the red channel in terms of overall scene brightness
- VH (cross-polarisation, weaker return) maps to green
- The blue channel's contribution is split equally between both, preserving the pretrained filter energy

The total weight energy per filter is conserved: `(R + 0.5B) + (G + 0.5B) = R + G + B`, so the network receives appropriately scaled inputs from the start without needing a large warm-up period.

### 7.3 build_resnet18_v2 — Dropout Regularisation (Cell `-iqMyPRw3lHm`)

```python
def build_resnet18_v2(pretrained_init=True, dropout=0.4):
    ...
    net.fc = nn.Sequential(
        nn.Dropout(dropout),   # dropout=0.4
        nn.Linear(512, 1),     # single logit for BCEWithLogitsLoss
    )
```

Dropout (p=0.4) before the final linear layer prevents the 512-dimensional feature vector from overfitting to training region characteristics — particularly important for the geographic generalisation task.

**Output:** A single raw logit. Applying `sigmoid` gives the probability of oil slick present. The threshold for the binary decision is 0.5.

---

## 8. Training Loop

> Training is the iterative process where the model looks at batches of labelled chips, makes predictions, measures how wrong it was (the loss), and updates its weights to do better next time. Several design choices here — how the loss is computed, how the learning rate changes over time, and when to stop training — each address a specific failure mode like overconfidence, unstable convergence, or overfitting. This section covers each of those choices and why they were made.

### 8.1 Loss Function

`BCEWithLogitsLoss` with **label smoothing** (ε=0.05):

```python
# Cell l01DkHKT3lHn
labels_smooth = labels * (1 - label_smoothing) + 0.5 * label_smoothing
loss = criterion(logits, labels_smooth)
```

Label smoothing replaces hard 0/1 targets with 0.025/0.975, preventing the model from becoming overconfident and improving calibration. The 0.5 factor means the smoothing target is the midpoint (0.5), nudging labels inward symmetrically.

### 8.2 Optimiser and Scheduler

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

- **AdamW** decouples weight decay from the adaptive learning rate update, giving more stable regularisation than vanilla Adam
- **CosineAnnealingLR** reduces the learning rate smoothly from `lr` to 0 over `T_max=80` epochs, allowing fine-grained convergence in later epochs without abrupt drops

### 8.3 Early Stopping and Checkpointing

```python
# Cell l01DkHKT3lHn
monitor = "f1"   # checkpoint when val F1 improves
patience = 20    # stop if no improvement for 20 consecutive epochs
```

The model state that achieved the best **validation F1** is saved to disk and restored at the end of training. F1 is used rather than loss because the dataset is nearly balanced and F1 directly reflects the classification quality; loss can improve even when F1 stagnates.

### 8.4 Training Hyperparameters (final configuration)

| Hyperparameter | Value |
|----------------|-------|
| Epochs (max) | 80 |
| Learning rate | 1e-4 |
| Weight decay | 5e-4 |
| Batch size | 32 |
| Dropout | 0.4 |
| Label smoothing | 0.05 |
| LR schedule | CosineAnnealingLR |
| Monitor | val F1 |
| Patience | 20 epochs |
| Gaussian noise σ | 0.02 |

### 8.5 Per-Epoch Metrics

After each epoch, the full validation set is evaluated without gradient computation:

```python
def evaluate(model, loader, device):
    # Returns: loss, accuracy, F1, AUROC, raw probs, labels, preds
    probs = torch.sigmoid(all_logits).numpy()
    preds = (probs >= 0.5).astype(int)
```

The training curve shows loss, F1, and AUROC over epochs (plotted in Cell `bKF5lzmz3lHo`).

---

## 9. Evaluation

> After training, we run the best saved model on the held-out test set — data it has never seen — to get an honest estimate of real-world performance. We evaluate under two different test conditions: the random split (same ocean regions as training) and the geographic split (Mediterranean only, never seen during training). Comparing the two tells us how well the model generalises beyond its training environment.

### 9.1 Metrics (Cell `Ridu_zfM3lHo`)

```python
def full_eval(model, loader, device, split_name):
    # Prints: Accuracy, F1, AUROC, full classification_report
    # Displays confusion matrix
```

Three complementary metrics are reported:

| Metric | What it measures |
|--------|-----------------|
| **Accuracy** | Fraction of correct predictions (useful baseline for balanced datasets) |
| **F1 Score** | Harmonic mean of precision and recall — robust to any residual class imbalance |
| **AUROC** | Area Under the ROC Curve — threshold-free discriminability; how well the model separates the two classes at any operating point |

F1 and AUROC are the primary metrics: F1 reflects real-world classification quality at the default 0.5 threshold; AUROC measures separability regardless of threshold, capturing how much headroom exists for tuning.

### 9.2 Geographic Split — Separate Retraining (Cell `j-o8eusx3lHp`)

The geographic split uses a **different training set** (non-Mediterranean regions), so it requires:
1. Recomputing channel statistics from the geographic training split
2. Training a fresh model from scratch with those statistics

This ensures the geographic evaluation genuinely tests out-of-distribution generalisation — the model has never seen Mediterranean ocean statistics during normalisation or training.

---

## 10. Results

> The numbers below tell us two things: how well the model works under ideal conditions (random split), and how much performance drops when it is deployed in a new region it has never seen (geographic split). The gap between these two scores is a proxy for real-world robustness.

From the latest run (`results/2026-06-13_15-36-23/`):

| Split | Accuracy | F1 | AUROC |
|-------|----------|----|-------|
| **Random** | 0.8036 | 0.8136 | 0.8573 |
| **Geographic (OOD)** | 0.6824 | 0.7117 | 0.8108 |

**Key observations:**

- The **random split** achieves strong performance (F1=0.814, AUROC=0.857), confirming that ResNet-18 with ImageNet weight transfer learns discriminative SAR features even without radar-specific pretraining.

- The **geographic split** shows a meaningful but moderate drop (F1 −0.10, AUROC −0.05). The AUROC remaining above 0.81 indicates the model has learned genuine signal features that transfer across regions, not just region-specific textures. The F1 gap reflects harder thresholding at 0.5 in a shifted distribution.

- The geographic AUROC (0.811) being close to the random AUROC (0.857) suggests that a calibrated threshold (different from 0.5) applied to Mediterranean data would recover much of the lost F1 — an area for operational tuning.

**Channel statistics (random training split):**
```
VV:  μ = −20.66 dB,  σ = 13.75 dB
VH:  μ = −25.76 dB,  σ = 15.93 dB
```

**Training convergence (random split):**
- Random split: best val F1 reached at epoch 24+, early stopped at epoch 37
- Geographic split: converged more slowly, early stopped at epoch 61

---

## 11. Failure Case Analysis

> Aggregate metrics tell us how often the model is wrong, but not *why*. Inspecting the most confidently wrong predictions — the cases where the model was sure and still got it wrong — reveals what visual patterns are confusing it and points toward specific improvements.

The `show_failures` function (Cell `gZeR21Gv3lHq`) identifies the **most confident wrong predictions** — cases where the model assigned high probability to the wrong class:

```python
wrong = np.where(preds != labels)[0]
confidence = np.abs(probs[wrong] - 0.5)   # distance from decision boundary
top_wrong = wrong[np.argsort(-confidence)[:n]]
```

Common failure patterns observed:
- **False positives:** Ship wakes or coastal roughness patterns that produce low-backscatter streaks resembling oil slicks
- **False negatives:** Thin or weathered slicks with backscatter similar to calm-water no-slick conditions
- **Geographic failures:** Mediterranean scenes with different ocean wind regimes producing unusual backscatter textures not seen during training

---

## 12. Standalone Training Script

[`train.py`](train.py) is a self-contained refactoring of the notebook that adds:

- **Timestamped run directories:** results saved to `results/YYYY-MM-DD_HH-MM-SS/` for experiment tracking
- **`train.log`:** all epoch output redirected to a log file (survives SSH disconnection)
- **`config.json`:** all hyperparameters saved at the start of each run
- **`channel_stats.json`:** normalisation statistics saved alongside results
- **Fully wired CONFIG dict:** all hyperparameters flow from a single dict into the model, loaders, and training functions

Run with:
```bash
conda activate oilslick
cd /mnt/data/home/sf2522/oilslick-detection
nohup python train.py > /dev/null 2>&1 &   # detach from terminal
tail -f results/<run_id>/train.log          # follow progress
```

---

## 13. TerraMind (GFM) Fine-tuning

[`train_terramind.py`](train_terramind.py) implements the second half of the comparison called
for by the project brief: a Geospatial Foundation Model variant fine-tuned on the same S1
inputs, evaluated against the ResNet-18 baseline under identical splits and metrics.

**Backbone:** [TerraMind-1.0-base](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base)
(IBM/ESA), loaded via `terratorch`'s `BACKBONE_REGISTRY` in single-modality `S1GRD` mode. The
backbone returns per-layer patch embeddings of shape `(B, 196, 768)`; the final layer is
mean-pooled over the 196 patch tokens and fed into a `Dropout(0.4) → Linear(768, 1)` head — the
same head shape as `build_resnet18_v2`, for a like-for-like comparison.

**What's reused from the CNN pipeline, unchanged:** split files, nodata masking, [−50, 10] dB
clipping, flip/rotation/noise augmentation, `BCEWithLogitsLoss` with label smoothing (0.05),
AdamW + `CosineAnnealingLR`, early stopping on val F1, and the evaluation/plotting code
(`evaluate`, `full_eval`, `show_failures`).

**What's different, and why:**
- **Normalisation statistics are fixed, not recomputed per split.** TerraMind's encoder was
  pretrained with its own S1GRD statistics (VV: μ=−12.599 σ=5.195, VH: μ=−20.293 σ=5.890 dB,
  from the TerraMesh pretraining corpus). Recomputing Welford stats per split — as done for
  ResNet-18 — would shift inputs off the distribution the encoder was pretrained on, especially
  damaging for the frozen linear-probe variant.
- **Two fine-tuning modes are trained per split**, both requested by the assignment brief:
  - `linear_probe` — backbone fully frozen (kept in `.eval()` regardless of the outer
    `train()`/`eval()` calls, so dropout/droppath inside it never activates), only the MLP head
    is trained, `lr=1e-3`, up to 50 epochs.
  - `full_finetune` — backbone unfrozen with a small discriminative LR (`2e-5`) below the head's
    (`1e-4`), reflecting that a pretrained encoder needs only gentle adaptation, up to 40 epochs.
- **Embedding dimension is inferred at build time** via a single dummy forward pass rather than
  hardcoded, so the script works unmodified if `MODEL_SIZE` is changed to `tiny`/`small`/`large`.

Run with (4 total training runs: 2 splits × 2 modes):
```bash
pip install "terratorch>=1.2.5"   # requires Python 3.11+
conda activate oilslick
cd /mnt/data/home/sf2522/oilslick-detection
nohup python train_terramind.py > /dev/null 2>&1 &
tail -f results/terramind_<run_id>/train.log
```

Outputs mirror `train.py`'s: `results/terramind_<run_id>/{config.json, train.log,
checkpoints/, plots/, results.json}`, with `results.json` reporting accuracy/F1/AUROC for
each (split, mode) pair so it can be diffed directly against the ResNet-18 `results.json`.

---

## 14. Project Structure

```
oilslick-detection/
├── oilslick_baseline_1.ipynb    # Main notebook (Sections 0–9)
├── train.py                     # Standalone training script — ResNet-18 baseline
├── train_terramind.py           # Standalone training script — TerraMind (GFM) fine-tuning
├── download_data.py             # Dataset download helper
│
├── data/data/OilSlick/
│   ├── metadata.csv             # 1363 samples: IDs, labels, coordinates
│   ├── images_s1/               # 981 GeoTIFF chips ({id}_s1.tif, 2-band, 224×224)
│   └── splits/
│       ├── random/              # train.txt / val.txt / test.txt
│       └── geographic/          # train.txt / val.txt / test.txt
│
├── results/
│   ├── 2026-06-13_15-36-23/     # Latest ResNet-18 run
│   │   ├── train.log            # Epoch-level training output
│   │   ├── config.json          # Hyperparameters used
│   │   ├── channel_stats.json   # VV/VH mean and std
│   │   ├── results.json         # Final test metrics (both splits)
│   │   ├── checkpoints/         # Best model weights per split
│   │   └── plots/               # Confusion matrices, training curves
│   └── terramind_<run_id>/      # TerraMind run — same layout, no channel_stats.json
│       ├── train.log
│       ├── config.json          # Includes fixed TerraMind S1GRD norm stats
│       ├── results.json         # Metrics per (split, mode) — linear_probe / full_finetune
│       ├── checkpoints/
│       └── plots/
│
└── checkpoints/                 # Root-level checkpoint copies
    ├── resnet18_random_split.pt
    └── resnet18_geographic_split.pt
```
