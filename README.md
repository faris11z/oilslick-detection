# Oil Slick Detection - ResNet-18 Baseline + TerraMind Linear Probe

Binary classification of **Sentinel-1 SAR satellite imagery** to detect the presence or absence of marine oil slicks. This project implements two models: a ResNet-18 baseline fully fine-tuned on 2-channel (VV/VH) SAR input, and a TerraMind-1.0-base geospatial foundation model used as a frozen feature extractor with a trained linear head. Both are evaluated under two split strategies: a standard random split and a geographically disjoint out-of-distribution split.

---

## Table of Contents

**ResNet-18 baseline**
1. [Problem Statement](#1-problem-statement)
2. [Dataset](#2-dataset)
3. [Data Exploration](#3-data-exploration)
4. [Preprocessing - ResNet-18](#4-preprocessing--resnet-18)
5. [Channel Statistics - Welford's Algorithm](#5-channel-statistics--welfords-algorithm)
6. [Dataset Class and DataLoaders](#6-dataset-class-and-dataloaders)
7. [Model Architecture - ResNet-18](#7-model-architecture--resnet-18)
8. [Training Loop - ResNet-18](#8-training-loop--resnet-18)
9. [Evaluation](#9-evaluation)
10. [Results](#10-results)
11. [Failure Case Analysis](#11-failure-case-analysis)
12. [Standalone Training Script - ResNet-18](#12-standalone-training-script--resnet-18)

**TerraMind linear probe**

13. [Model Architecture - TerraMind](#13-model-architecture--terramind)
14. [Preprocessing - TerraMind](#14-preprocessing--terramind)
15. [Feature Extraction Strategy](#15-feature-extraction-strategy)
16. [Training Loop - TerraMind Head](#16-training-loop--terramind-head)
17. [Standalone Training Script - TerraMind](#17-standalone-training-script--terramind)

**Ablation and outputs**

18. [ResNet-18 from Scratch - Ablation](#18-resnet-18-from-scratch--ablation)
19. [Comparison Plots](#19-comparison-plots)
20. [Project Structure](#20-project-structure)

---

## 1. Problem Statement

Marine oil slicks are a serious environmental hazard. Detecting them from satellite imagery enables rapid incident response. Sentinel-1 is a radar satellite (SAR - Synthetic Aperture Radar) that operates in all weather and lighting conditions, making it well-suited for operational maritime monitoring.

**Task:** Given a 224×224 pixel GeoTIFF chip with two radar polarisation channels (VV and VH), predict whether an oil slick is present (`label=1`) or absent (`label=0`).

**Why SAR?** Unlike optical sensors, SAR penetrates cloud cover and works at night. Oil slicks dampen ocean surface roughness, producing a distinctive low-backscatter signature visible in both VV and VH channels.

**Key challenge:** A model trained on one geographic region must generalise to unseen ocean areas and environmental conditions - tested via the geographic split.

---

## 2. Dataset

**Source:** WaterBench OilSlick dataset ([`metadata.csv`](data/data/OilSlick/metadata.csv))

The dataset contains 1363 annotated samples across four categories:
- `pos_*` - confirmed oil slick present (486 on disk)
- `neg_*` - confirmed no oil slick (484 on disk)
- `ext_pos_*` / `ext_neg_*` - extended set (11 on disk; excluded from training)

Each sample is a **2-band GeoTIFF** at 224×224 pixels:
- **Band 1: VV** - vertically transmitted, vertically received polarisation
- **Band 2: VH** - vertically transmitted, horizontally received polarisation

```
Cell 2 (notebook):
    BASE_DIR  = "/mnt/data/home/sf2522/oilslick-detection"
    OILSLICK  = os.path.join(BASE_DIR, "data", "data", "OilSlick")
    img_dir   = os.path.join(OILSLICK, "images_s1")

    tif_count = len([f for f in os.listdir(img_dir) if f.endswith(".tif")])
    # → TIF files: 981  (expected 1363)
```

**Note:** Only 981 of 1363 annotated samples are physically on disk - the remainder were not downloaded. The pipeline filters automatically to only use samples present on disk.

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

Chips are loaded with `rasterio` and displayed using a 2nd-98th percentile stretch to account for the heavy-tailed dB distribution:

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
- **Mean VV: −28.8 dB**, Std: 42.6 - extremely high std driven by outliers
- **Mean VH: −36.8 dB**, Std: 39.1 - similarly dominated by outliers
- **Minimum: −163** - a sentinel nodata value (land mask / missing data)
- **Maximum: ~1997** - extreme bright targets (ships, infrastructure)

These outliers would dominate z-score normalisation and squash the meaningful ocean signal, motivating the clipping step described next.

---

## 4. Preprocessing - ResNet-18

> Raw SAR values straight off the satellite are not suitable for a neural network. They contain missing data placeholders, extreme outliers from ships and coastlines, and wildly different value ranges between chips. This section describes how each chip is cleaned and standardised into a form the model can learn from - every step here is motivated by the data exploration findings above.

All preprocessing is applied inside `OilSlickDataset.__getitem__` in a deterministic, per-sample order:

```python
# Step 1 - Mask nodata sentinel (Cell wvdxlDVr3lHl)
chip[chip == -163.0] = 0.0

# Step 2 - Clip to physically meaningful dB range
chip[0] = np.clip(chip[0], -50, 10)   # VV channel
chip[1] = np.clip(chip[1], -50, 10)   # VH channel

# Step 3 - Z-score normalise per channel
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

## 5. Channel Statistics - Welford's Algorithm

> Z-score normalisation requires knowing the mean and standard deviation of the training data - but we can't load all 655 training chips into memory at once (each chip is 224×224×2 float32, and reading them all would require ~1 GB). This section describes how we compute exact statistics in a single pass through the data without ever holding more than one chip in memory at a time.

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

## 7. Model Architecture - ResNet-18

> The model is the part that actually learns to distinguish oil slicks from clean ocean. We use ResNet-18, a well-established image classification network, but it needs two modifications before it can work with SAR data: its input layer must be changed from 3 channels (RGB) to 2 channels (VV/VH), and its output layer must be changed from 1000 class scores (ImageNet) to a single oil-slick probability. This section explains how both changes are made without throwing away the pretrained knowledge the network already has.

### 7.1 Why ResNet-18?

ResNet-18 was chosen as the baseline for three reasons:
1. **Mature ImageNet pretraining** - feature extractors generalise well even to non-natural images
2. **Small enough to train fast** - 11.2M parameters, converges in under an hour on a single GPU
3. **Established benchmark** - widely used in remote sensing baselines, making results comparable

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

### 7.3 build_resnet18_v2 - Dropout Regularisation (Cell `-iqMyPRw3lHm`)

```python
def build_resnet18_v2(pretrained_init=True, dropout=0.4):
    ...
    net.fc = nn.Sequential(
        nn.Dropout(dropout),   # dropout=0.4
        nn.Linear(512, 1),     # single logit for BCEWithLogitsLoss
    )
```

Dropout (p=0.4) before the final linear layer prevents the 512-dimensional feature vector from overfitting to training region characteristics - particularly important for the geographic generalisation task.

**Output:** A single raw logit. Applying `sigmoid` gives the probability of oil slick present. The threshold for the binary decision is 0.5.

---

## 8. Training Loop - ResNet-18

> Training is the iterative process where the model looks at batches of labelled chips, makes predictions, measures how wrong it was (the loss), and updates its weights to do better next time. Several design choices here - how the loss is computed, how the learning rate changes over time, and when to stop training - each address a specific failure mode like overconfidence, unstable convergence, or overfitting. This section covers each of those choices and why they were made.

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
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=5e-4)
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
| Learning rate | 5e-5 |
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

> After training, we run the best saved model on the held-out test set - data it has never seen - to get an honest estimate of real-world performance. We evaluate under two different test conditions: the random split (same ocean regions as training) and the geographic split (Mediterranean only, never seen during training). Comparing the two tells us how well the model generalises beyond its training environment.

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
| **F1 Score** | Harmonic mean of precision and recall - robust to any residual class imbalance |
| **AUROC** | Area Under the ROC Curve - threshold-free discriminability; how well the model separates the two classes at any operating point |

F1 and AUROC are the primary metrics: F1 reflects real-world classification quality at the default 0.5 threshold; AUROC measures separability regardless of threshold, capturing how much headroom exists for tuning.

### 9.2 Geographic Split - Separate Retraining (Cell `j-o8eusx3lHp`)

The geographic split uses a **different training set** (non-Mediterranean regions), so it requires:
1. Recomputing channel statistics from the geographic training split
2. Training a fresh model from scratch with those statistics

This ensures the geographic evaluation genuinely tests out-of-distribution generalisation - the model has never seen Mediterranean ocean statistics during normalisation or training.

---

## 10. Results

> The numbers below tell us two things: how well the model works under ideal conditions (random split), and how much performance drops when it is deployed in a new region it has never seen (geographic split). The gap between these two scores is a proxy for real-world robustness.

**Full quantitative comparison (ResNet-18 vs TerraMind) is in [`RESULTS.md`](RESULTS.md)**, which covers per-split analysis, AUROC interpretation, and a discussion of when each model is the stronger choice.

### ResNet-18 - `results/2026-06-13_15-36-23/results.json`

| Split | Accuracy | F1 | AUROC |
|-------|----------|----|-------|
| **Random** | 0.8036 | 0.8136 | 0.8573 |
| **Geographic (OOD)** | 0.6824 | 0.7117 | 0.8108 |

### TerraMind - `results/2026-07-10_02-40-59/results.json`

| Split | Accuracy | F1 | AUROC |
|-------|----------|----|-------|
| **Random** | 0.7768 | 0.7934 | 0.8469 |
| **Geographic (OOD)** | 0.7095 | 0.7514 | 0.8014 |

**Key observations:**

- Under IID conditions (random split), ResNet-18 edges TerraMind on every metric (~+0.020 F1). This reflects the capacity difference: ResNet-18 had 11 M parameters fine-tuned end-to-end, while TerraMind trained only 770 parameters on top of a frozen backbone.

- Under distribution shift (geographic OOD), TerraMind outperforms ResNet-18 by **+0.040 F1** and **+0.027 accuracy**. The frozen backbone's SAR-specific pretraining generalises better to unseen ocean regions than a fine-tuned CNN that can overfit to training-region statistics.

- Both models maintain AUROC above 0.80 across all conditions, indicating that the underlying oil slick signal transfers geographically - the F1 gap at threshold 0.5 is partly a calibration issue that a tuned threshold would recover.

**Channel statistics (ResNet-18, random training split):**
```
VV:  μ = −20.66 dB,  σ = 13.75 dB
VH:  μ = −25.76 dB,  σ = 15.93 dB
```

**Training convergence:**
- ResNet-18 random: early stopped at epoch ~37
- ResNet-18 geographic: early stopped at epoch ~61
- TerraMind random head: early stopped at epoch 136, best val F1=0.768
- TerraMind geographic head: early stopped at epoch 162, best val F1=0.825

---

## 11. Failure Case Analysis

> Aggregate metrics tell us how often the model is wrong, but not *why*. Inspecting the most confidently wrong predictions - the cases where the model was sure and still got it wrong - reveals what visual patterns are confusing it and points toward specific improvements.

The `show_failures` function (Cell `gZeR21Gv3lHq`) identifies the **most confident wrong predictions** - cases where the model assigned high probability to the wrong class:

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

## 12. Standalone Training Script - ResNet-18

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

---

## 13. Model Architecture - TerraMind

### 13.1 What is TerraMind?

TerraMind-1.0-base is a geospatial foundation model developed jointly by IBM and ESA, available on HuggingFace at `ibm-esa-geospatial/TerraMind-1.0-base`. It is a Vision Transformer (ViT-Base) pretrained on large-scale, multi-modal Earth observation data including Sentinel-1 GRD, Sentinel-1 RTC, Sentinel-2 L2A, DEM, and LULC. Crucially for this project, it has a native Sentinel-1 GRD input modality - it was pretrained on the same sensor and polarisation bands (VV, VH) as the OilSlick dataset.

The architecture is a 12-layer ViT-Base transformer:
- **Hidden dimension:** 768
- **Attention heads:** 12
- **MLP ratio:** 4× with gated MLP (SwiGLU-style activation)
- **Patch size:** 16×16 pixels over a 224×224 input → 196 patch tokens per image
- **Total parameters:** ~300 M (frozen during training)

### 13.2 Linear Probing Approach

Rather than fine-tuning the full backbone, we use **linear probing**: the backbone weights are frozen entirely and a small classification head is trained on top of the extracted features. This is appropriate here because:

1. ~650 training samples is far too few to safely update 300 M parameters without catastrophic overfitting
2. TerraMind was pretrained on SAR data from the same sensor - the frozen features are already highly relevant for the task
3. It produces a clean comparison with ResNet-18: one model is fully supervised end-to-end, the other shows what pretrained SAR representations alone can achieve

**Trainable parameters:** 770 (Dropout + Linear(768→1)) out of ~300 M total backbone parameters.

### 13.3 Classification Head

```python
head = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(768, 1),   # single logit → BCEWithLogitsLoss
)
```

Input to the head is the **mean-pooled output of the last transformer layer**:

```python
out   = backbone(x)       # list of 12 layer outputs, each (B, 196, 768)
feats = out[-1].mean(dim=1)  # (B, 768) - global average pool over patch tokens
```

Mean pooling over all 196 patch tokens gives a single global representation of the chip. This is the standard linear probing protocol for ViT models and avoids any positional bias. The final layer (index −1, depth 12) contains the most abstract, task-agnostic representations and is the appropriate choice for classification probing.

Dropout (p=0.4) is applied before the linear layer to match the ResNet-18 head and reduce overconfident predictions - with only 770 parameters the main risk is overconfidence rather than overfitting in the traditional sense.

**Output:** A single raw logit. `sigmoid` gives the oil-slick probability; threshold is 0.5.

---

## 14. Preprocessing - TerraMind

### 14.1 Input modality

TerraMind is loaded with a single `untok_sen1grd@224` modality (untokenized Sentinel-1 GRD at 224×224). This is the correct native slot for 2-channel VV/VH input and ensures that the patch embedding weights used are those learned during pretraining on Sentinel-1 GRD data.

```python
backbone = BACKBONE_REGISTRY.build(
    "terramind_v1_base",
    pretrained=True,
    modalities=[{"untok_sen1grd@224": 2}],
)
```

The `untok` prefix means the data goes directly through a patch embedding layer (no discrete tokenisation). This is appropriate for backscatter values, which are continuous and meaningful as raw numbers.

### 14.2 Pixel cleaning

The same nodata masking and dB clipping as ResNet-18 is applied:

```python
chip[chip == -163.0] = 0.0          # mask nodata sentinel
chip[0] = np.clip(chip[0], -50, 10) # VV channel
chip[1] = np.clip(chip[1], -50, 10) # VH channel
```

### 14.3 Normalisation

Unlike ResNet-18, TerraMind does **not** compute statistics from the training split. Instead, fixed z-score statistics from TerraMind's own pretraining distribution are applied:

```python
_TM_MEAN = torch.tensor([-12.599, -20.293]).view(2, 1, 1)  # VV, VH in dB
_TM_STD  = torch.tensor([ 5.195,   5.890]).view(2, 1, 1)

chip = (chip - _TM_MEAN) / _TM_STD
```

These values (`v1_pretraining_mean["untok_sen1grd@224"]` and `v1_pretraining_std["untok_sen1grd@224"]`) are sourced directly from the terratorch library. Using the pretraining statistics is essential: the backbone's patch embedding weights are calibrated to this input scale and the model will produce meaningless representations if the input distribution shifts from what it saw during pretraining.

**Why not Welford statistics from training split?** The backbone is frozen - it has no mechanism to adapt to a different normalisation. The model has already converged to specific activations for inputs normalised as above; re-normalising to training-split statistics would push all inputs into an out-of-distribution range for the backbone.

### 14.4 Augmentation

During feature extraction, the same spatial augmentations as ResNet-18 are applied (horizontal flip, vertical flip, random 90° rotation, Gaussian noise σ=0.05). However, because features are pre-extracted and cached, augmentation only affects the single extraction pass - the head trains on those fixed augmented features. This provides modest diversity but does not give the same benefit as on-the-fly augmentation during training. A future improvement would be to extract features multiple times with different augmentations and pool them.

---

## 15. Feature Extraction Strategy

### Why pre-extract instead of running the backbone per batch?

With the backbone frozen, running it through the training loop is pure computation waste - the same input will always produce the same output. More practically, the GPU was fully occupied by other processes (only ~700 MiB free out of 44 GiB), making it impossible to fit a 300 M-parameter ViT in the training loop.

The solution is to run the backbone **once on CPU**, cache all features to disk, then train the head on those tensors - no backbone in the training loop at all.

```
Backbone (CPU, once)
    ├── rand_train → features/rand_train_feats.pt  (N, 768)
    ├── rand_val   → features/rand_val_feats.pt
    ├── rand_test  → features/rand_test_feats.pt
    ├── geo_train  → features/geo_train_feats.pt
    ├── geo_val    → features/geo_val_feats.pt
    └── geo_test   → features/geo_test_feats.pt

Head training (GPU, two separate runs)
    → loads cached (N, 768) tensors
    → trains Linear(768→1) only
```

After extraction the backbone is deleted (`del backbone`) and the GPU cache is cleared. This reduces GPU memory usage during head training to near zero.

**Caching:** if `features/` already contains `.pt` files from a previous run, extraction is skipped automatically. This makes re-runs fast when only the head hyperparameters are changed.

---

## 16. Training Loop - TerraMind Head

### 16.1 Loss function

`BCEWithLogitsLoss` with label smoothing ε=0.05, identical to ResNet-18:

```python
smooth = labels * (1 - 0.05) + 0.5 * 0.05   # → targets in [0.025, 0.975]
loss   = criterion(head(feats).squeeze(-1), smooth)
```

### 16.2 Optimiser and scheduler

```python
optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
```

The learning rate is higher than ResNet-18 (1e-3 vs 5e-5) because the head is initialised randomly and needs to reach a reasonable solution quickly. With only 770 parameters there is no risk of instability at this rate. Weight decay is slightly higher (1e-3 vs 5e-4) for the same reason - the small parameter count makes explicit regularisation less critical, but it stabilises training.

### 16.3 Training hyperparameters

| Hyperparameter | Value | Note |
|---|---|---|
| Epochs (max) | 200 | More than ResNet-18 because each epoch is ~1 ms - no cost to running longer |
| Learning rate | 1e-3 | Higher than ResNet-18; safe given the tiny head |
| Weight decay | 1e-3 | Slightly higher than ResNet-18 |
| Batch size | 256 | Large batch is fine - no image I/O, just matrix multiplications on cached tensors |
| Dropout | 0.4 | Matched to ResNet-18 for a fair comparison |
| Label smoothing | 0.05 | Identical to ResNet-18 |
| LR schedule | CosineAnnealingLR, T_max=200 | |
| Monitor | val F1 | Identical to ResNet-18 |
| Patience | 30 epochs | Longer than ResNet-18's 20 - head loss surface is noisy, needs more patience |

### 16.4 Early stopping and checkpointing

The best head state by validation F1 is checkpointed and restored at the end of training, identical protocol to ResNet-18. Only the head state (~3 KB) is saved, not the backbone.

Training convergence:
- Random split: stopped at epoch 136, best val F1=0.768
- Geographic split: stopped at epoch 162, best val F1=0.825

---

## 17. Standalone Training Script - TerraMind

[`run_terramind.py`](run_terramind.py) handles the full TerraMind pipeline end-to-end:

1. Builds the backbone and extracts features for all six splits (3 splits × 2 datasets) to `RUN_DIR/features/`
2. Deletes the backbone and clears GPU cache
3. Trains the linear head on random-split features, saves checkpoint and plots
4. Trains a fresh head on geographic-split features, saves checkpoint and plots
5. Loads pre-saved ResNet-18 results from `results/2026-06-13_15-36-23/results.json` and prints the full comparison table
6. Saves `results.json` and `histories.json` to the timestamped run directory

Run with:
```bash
conda activate oilslick
cd /mnt/data/home/sf2522/oilslick-detection
python run_terramind.py
```

Output structure per run:
```
results/YYYY-MM-DD_HH-MM-SS/
├── results.json         # final test metrics (both splits)
├── histories.json       # val F1 and AUROC per epoch (both heads)
├── features/            # cached (N, 768) tensors - skipped on re-run
│   ├── rand_train_feats.pt / rand_train_labels.pt
│   ├── rand_val_feats.pt   / rand_val_labels.pt
│   ├── rand_test_feats.pt  / rand_test_labels.pt
│   ├── geo_train_feats.pt  / geo_train_labels.pt
│   ├── geo_val_feats.pt    / geo_val_labels.pt
│   └── geo_test_feats.pt   / geo_test_labels.pt
├── checkpoints/
│   ├── terramind_random_best.pt
│   └── terramind_geographic_best.pt
└── plots/
    ├── training_curves_random.png
    ├── training_curves_geo.png
    ├── confusion_matrix_random.png
    └── confusion_matrix_geo.png
```

---

## 18. ResNet-18 from Scratch - Ablation

[`train_scratch.py`](train_scratch.py) trains ResNet-18 with **no pretrained weights** using an otherwise identical pipeline to `train.py`. Its purpose is to quantify how much of ResNet-18's performance comes from ImageNet initialisation versus the training data and pipeline.

Changes from `train.py`:

| Setting | `train.py` | `train_scratch.py` |
|---|---|---|
| Pretrained weights | ImageNet (torchvision) | None (`weights=None`) |
| conv1 init | RGB-blended pretrained weights | Kaiming uniform (PyTorch default) |
| Learning rate | 5e-5 | 1e-3 |
| Run dir suffix | _(none)_ | `_scratch` |

**Why a higher learning rate?** The fine-tuning LR of 5e-5 is calibrated for small updates on an already-converged network. A randomly initialised network needs much larger steps to escape the random weight regime; 1e-3 is the standard from-scratch LR for ResNet with AdamW.

Run with:
```bash
conda activate oilslick
cd /mnt/data/home/sf2522/oilslick-detection
python train_scratch.py
```

Results are saved to `results/YYYY-MM-DD_HH-MM-SS_scratch/`.

---

## 19. Comparison Plots

Poster-quality comparison figures are generated by [`make_comparison_plots.py`](make_comparison_plots.py) and saved to [`results/comparison_plots/`](results/comparison_plots/). Re-generate at any time:

```bash
conda activate oilslick
python make_comparison_plots.py
```

| File | What it shows |
|------|---------------|
| `fig1_f1_bar.png` | Grouped bar chart - F1 score for ResNet-18 and TerraMind side by side within each split (Random / Geographic). Shows which model wins on each split at a glance. |
| `fig2_ood_f1.png` | Slope chart - F1 score per model connected across splits (Random → Geographic). The slope direction and annotated Δ show the magnitude of OOD degradation for each model. |
| `fig3_f1_auroc_hue.png` | Two-panel bar chart - left panel F1, right panel AUROC, same grouped layout as fig1. Lets you compare both metrics in one view. |
| `fig4_confusion_matrices.png` | 2×2 grid of confusion matrices - rows are splits (Random, Geographic), columns are models (ResNet-18, TerraMind). All four rendered identically from saved checkpoints; numbers show raw counts. |
| `fig5_results_table.png` | Summary table - rows grouped by split, columns are Accuracy / F1 / AUROC. Within each split group the better model's values are bolded. Random rows shaded light blue, Geographic rows shaded light orange. |
| `fig6_dumbbell.png` | Dumbbell chart - one column per model; filled dot = Random F1, open dot = Geographic F1, connected by a vertical segment. Absolute F1 values labeled beside each dot; Δ labeled on the segment in bold. Shows individual scores and degradation gap simultaneously in a single compact view. |

**Color conventions used across all plots:**
- Light blue (`#6BAED6`) - ResNet-18
- Light orange (`#FC8D59`) - TerraMind

---

## 20. Project Structure

```
oilslick-detection/
├── oilslick_baseline_1.ipynb    # Main notebook (ResNet-18 Sections 0-9, TerraMind Section 10)
├── train.py                     # ResNet-18 standalone training script (ImageNet pretrained)
├── train_scratch.py             # ResNet-18 ablation - same pipeline, no pretrained weights
├── run_terramind.py             # TerraMind feature extraction + head training script
├── make_comparison_plots.py     # Generates all 5 comparison figures → results/comparison_plots/
├── download_data.py             # Dataset download helper
├── RESULTS.md                   # Quantitative results and comparison analysis
├── DEVLOG.md                    # Chronological development log
│
├── data/data/OilSlick/
│   ├── metadata.csv             # 1363 samples: IDs, labels, coordinates
│   ├── images_s1/               # 981 GeoTIFF chips ({id}_s1.tif, 2-band, 224×224)
│   └── splits/
│       ├── random/              # train.txt / val.txt / test.txt
│       └── geographic/          # train.txt / val.txt / test.txt
│
├── results/
│   ├── 2026-06-13_15-36-23/    # ResNet-18 run (pretrained)
│   │   ├── train.log            # Epoch-level training output
│   │   ├── config.json          # Hyperparameters used
│   │   ├── channel_stats.json   # VV/VH mean and std (Welford)
│   │   ├── results.json         # Final test metrics (both splits)
│   │   ├── checkpoints/         # Best model weights per split
│   │   └── plots/               # Confusion matrices, training curves, failure cases
│   │
│   ├── 2026-07-10_02-40-59/    # TerraMind run
│   │   ├── results.json         # Final test metrics (both splits)
│   │   ├── histories.json       # Val F1 / AUROC per epoch
│   │   ├── features/            # Cached (N, 768) backbone features
│   │   ├── checkpoints/         # Best head weights per split
│   │   └── plots/               # Confusion matrices, training curves
│   │
│   └── comparison_plots/        # Cross-model comparison figures (see Section 19)
│       ├── fig1_f1_bar.png      # F1 grouped bar chart
│       ├── fig2_ood_f1.png      # OOD slope chart
│       ├── fig3_f1_auroc_hue.png # F1 + AUROC side-by-side panels
│       ├── fig4_confusion_matrices.png  # 2×2 confusion matrix grid
│       ├── fig5_results_table.png       # Results summary table
│       └── fig6_dumbbell.png            # Dumbbell chart: F1 scores + degradation gap
│
└── checkpoints/                 # Root-level checkpoint copies
    ├── resnet18_random_split.pt
    └── resnet18_geographic_split.pt
```
