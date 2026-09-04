# Development Log - Oil Slick Detection

Chronological record of changes, fixes, and decisions made during development.

---

## 2026-07-09 / 2026-07-10 - TerraMind integration

### New files
- **`run_terramind.py`** - standalone training script for TerraMind linear probing (frozen backbone + small classification head). Trains on both random and geographic splits, saves metrics JSON, training curve PNGs, confusion matrix PNGs, and checkpoints to a timestamped `results/YYYY-MM-DD_HH-MM-SS/` directory. Loads pre-saved ResNet-18 results from `results/2026-06-13_15-36-23/results.json` for the final comparison table - no ResNet retraining needed.

### Notebook changes (`oilslick_baseline_1.ipynb`)
- Added **Section 10 - TerraMind (Frozen Backbone / Linear Probing)** cells:
  - Markdown header explaining the frozen backbone approach
  - `!pip install -q terratorch` install cell
  - `TerraMindClassifier` class definition (mirrors `run_terramind.py`)
  - `OilSlickDatasetRaw` + `make_terramind_loaders` dataset cell
  - Single display cell that reads results from the latest timestamped directory and renders training curves, confusion matrices, and a comparison table inline - no retraining inside the notebook

### Bugs fixed in `run_terramind.py`

#### 1. Wrong backbone registry name
- **Before:** `BACKBONE_REGISTRY.build("ibm-esa-geospatial/TerraMind-1.0-base")`
- **After:** `BACKBONE_REGISTRY.build("terramind_v1_base", pretrained=True, modalities=[{"untok_sen1grd@224": 2}])`
- **Why:** `BACKBONE_REGISTRY` uses terratorch's internal function names, not HuggingFace repo IDs. The correct key is `terramind_v1_base`. The `modalities` arg wires the model for 2-channel (VV/VH) S1 GRD input using the `untok_sen1grd@224` modality slot from MODALITY_INFO.

#### 2. Wrong forward-pass input key
- **Before:** `self.backbone({"S1GRD": x})` - key not found in `mod_name_mapping`, model would warn and raise "No valid inputs provided"
- **After:** `self.backbone(x)` - plain tensor auto-routed to the single registered modality (`untok_sen1grd@224`)

#### 3. Missing input normalisation
- **Before:** data clipped to [−50, 10] dB but not normalised
- **After:** z-score applied in `OilSlickDatasetRaw.__getitem__` using TerraMind v1 pretraining stats for `untok_sen1grd@224`:
  - mean: `[−12.599, −20.293]` dB (VV, VH)
  - std:  `[  5.195,   5.890]` dB (VV, VH)
- **Why:** `untok` modalities go through patch embedding only - no internal normalisation layer. Must match pretraining distribution manually.

#### 4. CUDA out of memory during training
- **Error:** `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.12 GiB. GPU has 698 MiB free` (two other processes occupied ~41 GiB of the 44 GiB card)
- **Fix:** switched to **feature pre-extraction** strategy:
  - Backbone runs once on **CPU** over all splits, producing `(N, 768)` feature tensors saved to `RUN_DIR/features/`
  - The training loop only trains a `Linear(768→1)` head on those cached tensors - backbone is never in the training loop
  - Backbone is `del`-ed after extraction and `torch.cuda.empty_cache()` called
  - Features are cached to disk so re-runs skip extraction
- **Side effects:** batch size for head training raised to 256 (tiny tensors), epochs raised to 200 with patience=30 since head-only training is much faster

### Infrastructure fixes
- Installed `c-compiler` via conda (`conda install -n oilslick -c conda-forge c-compiler`) to unblock `pip install terratorch` which requires gcc to build `stringzilla`.

### Architecture decision - linear probing vs fine-tuning
Chose **frozen backbone + linear head** (linear probing) over full fine-tuning because:
- ~650 training samples is too few to safely update ~300 M backbone parameters
- Produces a clean comparison with ResNet-18: same data, frozen pretrained features vs. trained-from-scratch CNN
- Consistent with project description wording ("fine-tuned with a small linear or MLP head")

Trainable parameters: ~770 (Dropout + Linear(768→1)) out of ~300 M total.

---

## 2026-07-10 - ResNet-18 from-scratch ablation script

Created `train_scratch.py` - identical pipeline to `train.py` with two changes:

1. **`pretrained_init=False`** - `models.resnet18(weights=None)`, no ImageNet weights loaded
2. **`_adapt_conv1_random`** - first conv initialised with Kaiming uniform (PyTorch default for Conv2d) instead of blending RGB pretrained weights; there is nothing to blend from
3. **Learning rate raised from 5e-5 → 1e-3** - the fine-tuning LR used in `train.py` is too small for a randomly initialised network; it would take hundreds of epochs to escape the random weight regime. 1e-3 is a standard from-scratch LR for ResNet with AdamW.

Run directory gets `_scratch` suffix so results don't collide with the pretrained run.

Everything else - preprocessing, Welford stats, augmentation, dropout, label smoothing, early stopping, plots, failure analysis - is identical to `train.py` to keep the comparison clean.

Run:
```bash
conda activate oilslick && python train_scratch.py
```

## 2026-07-10 - Results and pipeline documentation

Full results, analysis, and pipeline comparison written to dedicated files:
- Quantitative results and interpretation → `RESULTS.md`
- TerraMind pipeline details (preprocessing, architecture, training params) → `README.md` Sections 13-17

---

## 2026-06-13 - ResNet-18 baseline

- Trained ResNet-18 (2-channel input, pretrained ImageNet weights with first conv adapted) on random and geographic splits.
- Results saved to `results/2026-06-13_15-36-23/results.json`:
  - Random split:     Acc=0.8036  F1=0.8136  AUROC=0.8573
  - Geographic split: Acc=0.6824  F1=0.7117  AUROC=0.8108
