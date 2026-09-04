"""
Download and extract the OilSlick subset of WaterBench from HuggingFace.
Run from: /mnt/data/home/sf2522/oilslick-detection/
"""

import os
import subprocess
from huggingface_hub import snapshot_download

TARGET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OILSLICK_DIR = os.path.join(TARGET_DIR, "OilSlick")

print("Downloading dataset from HuggingFace...")
snapshot_download(
    repo_id="ayushprd/WaterBench",
    repo_type="dataset",
    allow_patterns=[
        "data/OilSlick/OilSlick-images_s1-00.tar",
        "data/OilSlick/OilSlick-images_s1-01.tar",
        "data/OilSlick/metadata.csv",
        "data/OilSlick/metadata.json",
        "data/OilSlick/splits/random/*.txt",
        "data/OilSlick/splits/geographic/*.txt",
    ],
    local_dir=TARGET_DIR,
    max_workers=1,  # serial downloads to avoid brotli/httpx crashes
)
print("Download complete.")

print("Extracting archives...")
for tar_file in ["OilSlick-images_s1-00.tar", "OilSlick-images_s1-01.tar"]:
    tar_path = os.path.join(OILSLICK_DIR, tar_file)
    if not os.path.exists(tar_path):
        print(f"  WARNING: {tar_file} not found, skipping.")
        continue
    print(f"  Extracting {tar_file}...")
    subprocess.run(["tar", "xf", tar_path, "-C", OILSLICK_DIR], check=True)
    print(f"  Done: {tar_file}")

print("Verifying extraction...")
img_dir = os.path.join(OILSLICK_DIR, "images_s1")
if os.path.exists(img_dir):
    tif_count = len([f for f in os.listdir(img_dir) if f.endswith(".tif")])
    print(f"TIF files found: {tif_count}  (expected 1363)")
else:
    print("WARNING: images_s1/ directory not found after extraction.")
