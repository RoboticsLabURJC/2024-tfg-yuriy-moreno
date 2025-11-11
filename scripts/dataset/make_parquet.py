#!/usr/bin/env python3
import os, random, shutil
from pathlib import Path
import pandas as pd

# === CONFIGURACIÓN ===
PINAR = Path("/home/yuriy/Universidad/2024-tfg-yuriy-moreno/scripts/dataset/pinar_train")
RGB_DIR = PINAR / "rgb"
SEG_DIR = PINAR / "segmentation_idx"
RGB_PREFIX = "rgb_"
SEG_PREFIX = "segmentation_idx_"
EXT = ".png"
SPLITS = {"train": 0.6, "val": 0.2, "test": 0.2}
SEED = 42
MOVE_FILES = True  # True = mover, False = copiar

# === RECOGER ARCHIVOS ===
rgb_files = sorted(RGB_DIR.glob(f"{RGB_PREFIX}*{EXT}"))
seg_files = sorted(SEG_DIR.glob(f"{SEG_PREFIX}*{EXT}"))

rgb_ids = [f.stem.replace(RGB_PREFIX, "") for f in rgb_files]
seg_ids = [f.stem.replace(SEG_PREFIX, "") for f in seg_files]
common_ids = sorted(set(rgb_ids).intersection(seg_ids))

if not common_ids:
    raise RuntimeError("⚠️ No se encontraron pares RGB / Segmentación con IDs coincidentes")

# === DIVISIÓN 60/20/20 ===
random.seed(SEED)
random.shuffle(common_ids)
n = len(common_ids)
n_train = int(0.6 * n)
n_val = int(0.2 * n)
splits = {
    "train": common_ids[:n_train],
    "val": common_ids[n_train:n_train+n_val],
    "test": common_ids[n_train+n_val:]
}

# === CREAR CARPETAS DESTINO ===
for s in splits:
    (PINAR / s / "rgb").mkdir(parents=True, exist_ok=True)
    (PINAR / s / "segmentation_idx").mkdir(parents=True, exist_ok=True)

# === MOVER/COPIAR Y CONSTRUIR DATAFRAME ===
rows = []
for split_name, ids in splits.items():
    for img_id in ids:
        rgb_name = f"{RGB_PREFIX}{img_id}{EXT}"
        seg_name = f"{SEG_PREFIX}{img_id}{EXT}"

        src_rgb = RGB_DIR / rgb_name
        src_seg = SEG_DIR / seg_name
        dst_rgb = PINAR / split_name / "rgb" / rgb_name
        dst_seg = PINAR / split_name / "segmentation_idx" / seg_name

        if MOVE_FILES:
            shutil.move(src_rgb, dst_rgb)
            shutil.move(src_seg, dst_seg)
        else:
            shutil.copy2(src_rgb, dst_rgb)
            shutil.copy2(src_seg, dst_seg)

        rows.append({
            "id": img_id,
            "image": str(dst_rgb.relative_to(PINAR)),
            "label": str(dst_seg.relative_to(PINAR)),
            "split": split_name
        })

# === GUARDAR DATASET.PARQUET Y CSV ===
df = pd.DataFrame(rows).sort_values(["split", "id"]).reset_index(drop=True)
out_parquet = PINAR / "dataset.parquet"
out_csv = PINAR / "dataset.csv"
df.to_parquet(out_parquet, index=False)
df.to_csv(out_csv, index=False)

print(f"✅ División completa ({n} pares totales):")
for s, ids in splits.items():
    print(f"  {s.capitalize():<5}: {len(ids)} imágenes")
print(f"📦 Guardado:")
print(f"  - {out_parquet}")
print(f"  - {out_csv}")
