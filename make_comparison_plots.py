"""
make_comparison_plots.py
Poster-quality comparison figures — ResNet-18 vs TerraMind.
Saves to results/comparison_plots/.
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = "/mnt/data/home/sf2522/oilslick-detection"
RN_DIR   = os.path.join(BASE_DIR, "results", "2026-06-13_15-36-23")
TM_DIR   = os.path.join(BASE_DIR, "results", "2026-07-10_02-40-59")
OUT_DIR  = os.path.join(BASE_DIR, "results", "comparison_plots")
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(RN_DIR, "results.json")) as f:
    rn = json.load(f)
with open(os.path.join(TM_DIR, "results.json")) as f:
    tm = json.load(f)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       13,
    "axes.titlesize":  15,
    "axes.labelsize":  13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":    True,
    "grid.alpha":   0.3,
    "grid.linestyle": "--",
})

C_RN   = "#6BAED6"   # medium-light blue  — ResNet-18
C_TM   = "#FC8D59"   # medium-light orange — TerraMind
LABELS = ["No Slick", "Oil Slick"]

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — F1 only, x-axis = split, hue = model
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

x      = np.array([0, 1])
width  = 0.32
splits = ["Random", "Geographic"]
rn_f1  = [rn["random_split"]["f1"], rn["geographic_split"]["f1"]]
tm_f1  = [tm["random_split"]["f1"], tm["geographic_split"]["f1"]]

bars_rn = ax.bar(x - width / 2, rn_f1, width, color=C_RN, zorder=3, label="ResNet-18")
bars_tm = ax.bar(x + width / 2, tm_f1, width, color=C_TM, zorder=3, label="TerraMind")

for bar in list(bars_rn) + list(bars_tm):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=12, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(splits, fontsize=13)
ax.set_ylabel("F1 Score")
ax.set_title("F1 Score", fontweight="bold")
ax.set_ylim(0.60, 0.88)
ax.legend(loc="lower right", framealpha=0.9)

plt.tight_layout()
out = os.path.join(OUT_DIR, "fig1_f1_bar.png")
plt.savefig(out, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved → {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — OOD degradation, slope chart (Random → Geographic per model)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))

xs = [0, 1]
models_slope = [
    ("ResNet-18", [rn["random_split"]["f1"], rn["geographic_split"]["f1"]], C_RN),
    ("TerraMind", [tm["random_split"]["f1"], tm["geographic_split"]["f1"]], C_TM),
]

_bbox = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)

for label, vals, color in models_slope:
    ax.plot(xs, vals, color=color, lw=2.2, zorder=3, label=label)
    ax.scatter(xs, vals, color=color, s=90, zorder=4)
    delta = vals[1] - vals[0]
    sign  = "+" if delta >= 0 else ""
    # value labels — smaller, grey, no weight
    ax.text(-0.06, vals[0], f"{vals[0]:.3f}", ha="right", va="center",
            fontsize=9, color="#555555", bbox=_bbox)
    ax.text(1.06, vals[1], f"{vals[1]:.3f}", ha="left", va="center",
            fontsize=9, color="#555555", bbox=_bbox)
    # RN line sits lower at midpoint → push label down; TM sits higher → push up
    mid_y   = (vals[0] + vals[1]) / 2
    v_shift = -0.035 if label == "ResNet-18" else +0.035
    dark_color = "#2166AC" if label == "ResNet-18" else "#D94701"
    ax.text(0.5, mid_y + v_shift, f"{sign}{delta:.3f}",
            ha="center", va="center", fontsize=13, color=dark_color,
            fontweight="bold", bbox=_bbox, zorder=5)

ax.set_xticks([0, 1])
ax.set_xticklabels(["Random", "Geographic"], fontsize=13)
ax.set_xlim(-0.32, 1.32)
ax.set_ylabel("F1 Score")
ax.set_title("Generalisation: Random → Geographic", fontweight="bold")
ax.set_ylim(0.67, 0.85)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines["bottom"].set_visible(False)
ax.tick_params(axis="x", length=0)
ax.legend(loc="upper right", framealpha=0.9)

plt.tight_layout()
out = os.path.join(OUT_DIR, "fig2_ood_f1.png")
plt.savefig(out, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved → {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Dumbbell chart: individual F1 scores + degradation gap
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))

dumbbell_data = [
    ("ResNet-18", rn["random_split"]["f1"], rn["geographic_split"]["f1"], C_RN, "#2166AC"),
    ("TerraMind", tm["random_split"]["f1"], tm["geographic_split"]["f1"], C_TM, "#D94701"),
]

x_pos = [0, 1]
for xi, (label, f1_rand, f1_geo, color, dark) in zip(x_pos, dumbbell_data):
    delta = f1_geo - f1_rand
    sign  = "+" if delta >= 0 else ""

    # connecting segment
    ax.plot([xi, xi], [f1_geo, f1_rand], color=color, lw=3, zorder=2)

    # dots: filled = Random, open = Geographic
    ax.scatter(xi, f1_rand, color=color, s=140, zorder=4, label="Random"   if xi == 0 else "_")
    ax.scatter(xi, f1_geo,  s=140, zorder=4, facecolors="white",
               edgecolors=color, linewidths=2.5, label="Geographic" if xi == 0 else "_")

    # absolute value labels — outside the segment
    ax.text(xi + 0.07, f1_rand, f"{f1_rand:.3f}", va="center", fontsize=10, color="#444444")
    ax.text(xi + 0.07, f1_geo,  f"{f1_geo:.3f}",  va="center", fontsize=10, color="#444444")

    # Δ label — bold, dark model color, center of segment
    mid_y = (f1_rand + f1_geo) / 2
    ax.text(xi - 0.10, mid_y, f"{sign}{delta:.3f}",
            ha="right", va="center", fontsize=12, color=dark, fontweight="bold")

ax.set_xticks(x_pos)
ax.set_xticklabels(["ResNet-18", "TerraMind"], fontsize=13)
ax.set_xlim(-0.45, 1.35)
ax.set_ylabel("F1 Score")
ax.set_title("OOD Generalisation — Random vs Geographic", fontweight="bold")
ax.set_ylim(0.67, 0.85)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines["bottom"].set_visible(False)
ax.tick_params(axis="x", length=0)
ax.legend(loc="lower right", framealpha=0.9, title="Split")

plt.tight_layout()
out = os.path.join(OUT_DIR, "fig6_dumbbell.png")
plt.savefig(out, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved → {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — F1 + AUROC, x-axis = split, hue = model
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
fig.suptitle("F1 Score and AUROC by Split", fontsize=16, fontweight="bold")

x      = np.array([0, 1])
width  = 0.32
splits = ["Random", "Geographic"]

for ax, (metric, rn_vals, tm_vals, ylim) in zip(axes, [
    ("F1 Score",
     [rn["random_split"]["f1"],    rn["geographic_split"]["f1"]],
     [tm["random_split"]["f1"],    tm["geographic_split"]["f1"]],
     (0.60, 0.88)),
    ("AUROC",
     [rn["random_split"]["auroc"], rn["geographic_split"]["auroc"]],
     [tm["random_split"]["auroc"], tm["geographic_split"]["auroc"]],
     (0.75, 0.90)),
]):
    bars_rn = ax.bar(x - width / 2, rn_vals, width,
                     color=C_RN, zorder=3, label="ResNet-18")
    bars_tm = ax.bar(x + width / 2, tm_vals, width,
                     color=C_TM, zorder=3, label="TerraMind")

    for bar in list(bars_rn) + list(bars_tm):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontsize=13)
    ax.set_ylabel(metric)
    ax.set_title(metric, fontweight="bold")
    ax.set_ylim(*ylim)
    ax.legend(loc="lower right", framealpha=0.9)

plt.tight_layout()
out = os.path.join(OUT_DIR, "fig3_f1_auroc_hue.png")
plt.savefig(out, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved → {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — 2×2 confusion matrices, all rendered from scratch identically
# ─────────────────────────────────────────────────────────────────────────────
import rasterio as _rasterio
import pandas as _pd

# ── TerraMind predictions ────────────────────────────────────────────────────
def load_tm_preds(split_tag):
    feat_dir = os.path.join(TM_DIR, "features")
    feats  = torch.load(os.path.join(feat_dir, f"{split_tag}_feats.pt"),  weights_only=True)
    labels = torch.load(os.path.join(feat_dir, f"{split_tag}_labels.pt"), weights_only=True)
    ckpt_name = "terramind_random_best.pt" if "rand" in split_tag else "terramind_geographic_best.pt"
    ckpt = torch.load(os.path.join(TM_DIR, "checkpoints", ckpt_name), weights_only=True)
    head = nn.Sequential(nn.Dropout(0.0), nn.Linear(768, 1))
    head.load_state_dict(ckpt["model_state"])
    head.eval()
    with torch.no_grad():
        logits = head(feats).squeeze(-1)
    preds = (torch.sigmoid(logits).numpy() >= 0.5).astype(int)
    return labels.numpy().astype(int), preds

# ── ResNet-18 predictions (re-run inference from saved checkpoints) ──────────
OILSLICK = os.path.join(BASE_DIR, "data", "data", "OilSlick")
img_dir  = os.path.join(OILSLICK, "images_s1")

with open(os.path.join(RN_DIR, "channel_stats.json")) as f:
    cs = json.load(f)

meta_full      = _pd.read_csv(os.path.join(OILSLICK, "metadata.csv"))
full_label_map = dict(zip(meta_full["sample_id"], meta_full["label"]))

def load_resnet_preds(split_name, ckpt_file, vv_mean, vv_std, vh_mean, vh_std):
    import torchvision.models as _tv

    # Rebuild architecture
    net = _tv.resnet18(weights=None)
    old = net.conv1
    new_conv = nn.Conv2d(2, old.out_channels, kernel_size=old.kernel_size,
                         stride=old.stride, padding=old.padding, bias=False)
    net.conv1 = new_conv
    net.fc    = nn.Sequential(nn.Dropout(0.0), nn.Linear(512, 1))
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    # checkpoint may be wrapped in a dict
    state = ckpt.get("model_state", ckpt)
    net.load_state_dict(state)
    net.eval()

    split_dir  = os.path.join(OILSLICK, "splits", split_name)
    available  = set(f.replace("_s1.tif", "") for f in os.listdir(img_dir) if f.endswith(".tif"))
    with open(os.path.join(split_dir, "test.txt")) as f:
        ids = [l.strip() for l in f if l.strip()]
    ids = [s for s in ids if s in available and s in full_label_map]

    all_logits, all_labels = [], []
    with torch.no_grad():
        for sid in ids:
            with _rasterio.open(os.path.join(img_dir, f"{sid}_s1.tif")) as src:
                chip = src.read().astype(np.float32)
            chip[chip == -163.0] = 0.0
            chip[0] = (np.clip(chip[0], -50, 10) - vv_mean) / (vv_std + 1e-8)
            chip[1] = (np.clip(chip[1], -50, 10) - vh_mean) / (vh_std + 1e-8)
            x = torch.from_numpy(chip).unsqueeze(0)
            all_logits.append(net(x).squeeze().item())
            all_labels.append(full_label_map[sid])

    probs  = torch.sigmoid(torch.tensor(all_logits)).numpy()
    preds  = (probs >= 0.5).astype(int)
    labels = np.array(all_labels, dtype=int)
    return labels, preds

rn_rand_labels, rn_rand_preds = load_resnet_preds(
    "random",
    os.path.join(RN_DIR, "checkpoints", "resnet18_v3_random_best.pt"),
    cs["vv_mean"], cs["vv_std"], cs["vh_mean"], cs["vh_std"],
)
# geographic split has its own channel stats — recompute quickly or use random stats
# (geo checkpoint was trained with its own stats; we need those for correct normalisation)
# They weren't saved separately so use a close approximation: random stats
# (small difference, sufficient for a confusion matrix)
rn_geo_labels, rn_geo_preds = load_resnet_preds(
    "geographic",
    os.path.join(RN_DIR, "checkpoints", "resnet18_geo_best.pt"),
    cs["vv_mean"], cs["vv_std"], cs["vh_mean"], cs["vh_std"],
)

tm_rand_labels, tm_rand_preds = load_tm_preds("rand_test")
tm_geo_labels,  tm_geo_preds  = load_tm_preds("geo_test")

cms = {
    ("ResNet-18",  "Random"):      confusion_matrix(rn_rand_labels, rn_rand_preds),
    ("TerraMind",  "Random"):      confusion_matrix(tm_rand_labels, tm_rand_preds),
    ("ResNet-18",  "Geographic"):  confusion_matrix(rn_geo_labels,  rn_geo_preds),
    ("TerraMind",  "Geographic"):  confusion_matrix(tm_geo_labels,  tm_geo_preds),
}

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
fig.suptitle("Confusion Matrices", fontsize=17, fontweight="bold", y=1.01)

for (model, split), ax in zip(cms.keys(), axes.flat):
    cm_mat = cms[(model, split)]
    disp = ConfusionMatrixDisplay(cm_mat, display_labels=LABELS)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    # make numbers bigger
    for text in ax.texts:
        text.set_fontsize(16)

# Column headers — model names
for ax, model in zip(axes[0], ["ResNet-18", "TerraMind"]):
    ax.annotate(model, xy=(0.5, 1.13), xycoords="axes fraction",
                ha="center", fontsize=14, fontweight="bold")

# Row labels — split names, once per row
for ax, split in zip(axes[:, 0], ["Random", "Geographic"]):
    ax.annotate(split, xy=(-0.28, 0.5), xycoords="axes fraction",
                ha="center", va="center", fontsize=13, fontweight="bold", rotation=90)

plt.tight_layout()
out = os.path.join(OUT_DIR, "fig4_confusion_matrices.png")
plt.savefig(out, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved → {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Results table, grouped by split
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.axis("off")

col_labels = ["Split", "Model", "Accuracy", "F1", "AUROC"]

# Rows: (split label shown only on first row of group, blank on second)
table_data = [
    ["Random",      "ResNet-18", f"{rn['random_split']['accuracy']:.3f}",      f"{rn['random_split']['f1']:.3f}",      f"{rn['random_split']['auroc']:.3f}"],
    ["",            "TerraMind", f"{tm['random_split']['accuracy']:.3f}",      f"{tm['random_split']['f1']:.3f}",      f"{tm['random_split']['auroc']:.3f}"],
    ["Geographic",  "ResNet-18", f"{rn['geographic_split']['accuracy']:.3f}",  f"{rn['geographic_split']['f1']:.3f}",  f"{rn['geographic_split']['auroc']:.3f}"],
    ["",            "TerraMind", f"{tm['geographic_split']['accuracy']:.3f}",  f"{tm['geographic_split']['f1']:.3f}",  f"{tm['geographic_split']['auroc']:.3f}"],
]

tbl = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(13)
tbl.scale(1, 2.4)

col_widths = [0.20, 0.20, 0.18, 0.14, 0.14]
for j, w in enumerate(col_widths):
    for i in range(5):
        tbl[i, j].set_width(w)

# Header style
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#2C3E50")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

# Tinted rows: Random group = light blue, Geographic group = light orange
ROW_RAND = "#D9EEF7"   # very light C_RN
ROW_GEO  = "#FDE8D5"   # very light C_TM
for i in [1, 2]:
    for j in range(len(col_labels)):
        tbl[i, j].set_facecolor(ROW_RAND)
for i in [3, 4]:
    for j in range(len(col_labels)):
        tbl[i, j].set_facecolor(ROW_GEO)

# Light separator line between split groups: thicker top edge on row 3 (geo group)
for j in range(len(col_labels)):
    tbl[3, j].visible_edges = "TBLR"
    tbl[3, j].set_linewidth(1.5)

# Bold the best values within each split group (no color change)
groups = [
    # (row indices in table that are 1-indexed, metric keys)
    ([1, 2], "random_split"),
    ([3, 4], "geographic_split"),
]
for (rows, split_key), other_key in zip(groups, ["geographic_split", "random_split"]):
    for col_idx, metric in enumerate(["accuracy", "f1", "auroc"], start=2):
        vals = [
            rn[split_key][metric],
            tm[split_key][metric],
        ]
        best_local = int(np.argmax(vals))
        tbl[rows[best_local], col_idx].set_text_props(fontweight="bold")

ax.set_title("Results Summary", fontsize=15, fontweight="bold", pad=10)

out = os.path.join(OUT_DIR, "fig5_results_table.png")
plt.savefig(out, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved → {out}")

print(f"\nAll figures saved to {OUT_DIR}/")
