import cv2
import numpy as np
import pandas as pd
import os
import shutil
from tqdm import tqdm

print("--- Memulai Augmentasi Sinkron Gambar + Mask ---")

# === [1] KONFIGURASI PATH ===
BASE_PATH = r"D:\TEKOM THINGS\SEMESTER 5\PCD\TUGAS AKHIR"

CSV_METADATA_INPUT = os.path.join(BASE_PATH, "dataset_hasil_pipeline.csv")
OUTPUT_IMG = os.path.join(BASE_PATH, "DATASET_AUGMENTED_FINAL")
OUTPUT_MASK = os.path.join(BASE_PATH, "DATASET_MASK_AUGMENTED")
CSV_METADATA_OUTPUT = os.path.join(BASE_PATH, "metadata_augmented_sinkron.csv")

for path in [OUTPUT_IMG, OUTPUT_MASK]:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
print(f"Folder output dibuat:\n{OUTPUT_IMG}\n{OUTPUT_MASK}")

# === [2] FUNGSI AUGMENTASI ===
def aug_flip(img): return cv2.flip(img, 1)
def aug_rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
def aug_brightness(img, factor):
    img_float = img.astype(float) * factor
    return np.clip(img_float, 0, 255).astype(np.uint8)
def aug_contrast(img, alpha=1.25, beta=0): return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
def aug_gamma(img, gamma=1.3):
    inv = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)
def aug_blur(img): return cv2.GaussianBlur(img, (5,5), 0)

# untuk mask: hanya spatial transforms (flip, rotate)
def aug_mask_flip(mask): return cv2.flip(mask, 1)
def aug_mask_rotate(mask, angle):
    h, w = mask.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    return cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

# === [3] PROSES UTAMA ===
df = pd.read_csv(CSV_METADATA_INPUT)
if 'split' not in df.columns or 'kelas' not in df.columns:
    raise ValueError("CSV pipeline harus memiliki kolom 'split' dan 'kelas'!")

new_records = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    nama_file = row['nama_file']
    kelas = row['kelas']
    split = row['split']

    path_img = row['path_final_morfologi']
    path_mask = row['path_mask_hole_filling']

    if not os.path.exists(path_img) or not os.path.exists(path_mask):
        print(f"[SKIP] File tidak ditemukan: {nama_file}")
        continue

    img = cv2.imread(path_img)
    mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    base, ext = os.path.splitext(nama_file)

    def save_pair(img_variant, mask_variant, suffix):
        new_name_img = f"{base}_{suffix}{ext}"
        new_name_mask = f"{base}_{suffix}_mask.png"
        out_img = os.path.join(OUTPUT_IMG, new_name_img)
        out_mask = os.path.join(OUTPUT_MASK, new_name_mask)
        cv2.imwrite(out_img, img_variant)
        cv2.imwrite(out_mask, mask_variant)
        new_records.append({
            "nama_file_baru": new_name_img,
            "nama_mask_baru": new_name_mask,
            "kelas": kelas,
            "split": split,
            "path_img": out_img,
            "path_mask": out_mask
        })

    if split == "Latih":
        # augment semua pasangan
        save_pair(img, mask, "original")
        save_pair(aug_flip(img), aug_mask_flip(mask), "flip")
        save_pair(aug_brightness(img, 1.2), mask, "bright")
        save_pair(aug_brightness(img, 0.8), mask, "dark")
        save_pair(aug_contrast(img, 1.25), mask, "contrast")
        save_pair(aug_gamma(img, 1.3), mask, "gamma")
        save_pair(aug_blur(img), mask, "blur")
        save_pair(aug_rotate(img, 7), aug_mask_rotate(mask, 7), "rotplus")
        save_pair(aug_rotate(img, -7), aug_mask_rotate(mask, -7), "rotmin")

    else:  # Validasi/Uji
        save_pair(img, mask, "original")

df_out = pd.DataFrame(new_records)
df_out.to_csv(CSV_METADATA_OUTPUT, index=False, encoding="utf-8-sig")

print("\n✅ Augmentasi sinkron selesai!")
print(f"Metadata disimpan di: {CSV_METADATA_OUTPUT}")
print(f"Gambar: {OUTPUT_IMG}")
print(f"Mask: {OUTPUT_MASK}")
