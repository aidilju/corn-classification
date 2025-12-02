import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from skimage.feature import graycomatrix as greycomatrix, graycoprops as greycoprops
from skimage.measure import regionprops, label

print("--- Memulai Ekstraksi Fitur v16 (Sinkron Gambar + Mask) ---")

# === [1] KONFIGURASI PATH ===
BASE_PATH = r"D:\TEKOM THINGS\SEMESTER 5\PCD\TUGAS AKHIR"

CSV_METADATA_INPUT = os.path.join(BASE_PATH, "metadata_augmented_sinkron.csv")
CSV_OUTPUT_FITUR = os.path.join(BASE_PATH, "dataset_fitur_final_v16.csv")

# === [2] DEFINISI FUNGSI EKSTRAKSI ===

def extract_color_features(image):
    """Ekstraksi Fitur Warna (HSV)"""
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        mask = v > 0
        if not np.any(mask):
            raise ValueError("Masker kosong")
        return {
            "h_mean": np.mean(h[mask]), "h_std": np.std(h[mask]),
            "s_mean": np.mean(s[mask]), "s_std": np.std(s[mask]),
            "v_mean": np.mean(v[mask]), "v_std": np.std(v[mask]),
        }
    except Exception:
        return {k: 0 for k in ["h_mean", "h_std", "s_mean", "s_std", "v_mean", "v_std"]}


def extract_texture_features(image_color):
    """Ekstraksi Fitur Tekstur (GLCM)"""
    try:
        gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
        mask = gray > 0
        if not np.any(mask):
            raise ValueError("Masker kosong")

        # Skala ke 16 level
        gray_scaled = (gray.astype(float) / 255.0 * 15).astype(np.uint8) + 1
        gray_scaled[~mask] = 0

        glcm = greycomatrix(
            gray_scaled,
            distances=[5],
            angles=[0],
            levels=17,
            symmetric=True,
            normed=True,
        )
        glcm = glcm[1:, 1:]
        glcm_sum = np.sum(glcm)
        if glcm_sum == 0:
            raise ValueError("GLCM kosong")
        glcm = glcm / glcm_sum

        return {
            "contrast": greycoprops(glcm, "contrast")[0, 0],
            "correlation": greycoprops(glcm, "correlation")[0, 0],
            "energy": greycoprops(glcm, "energy")[0, 0],
            "homogeneity": greycoprops(glcm, "homogeneity")[0, 0],
        }
    except Exception:
        return {k: 0 for k in ["contrast", "correlation", "energy", "homogeneity"]}


def extract_shape_features(mask_image_grayscale):
    """Ekstraksi Fitur Bentuk (Shape)"""
    try:
        if mask_image_grayscale is None or mask_image_grayscale.size == 0:
            raise ValueError("Masker kosong")

        # Pastikan biner
        mask_bool = mask_image_grayscale > 0
        label_img = label(mask_bool)
        props = regionprops(label_img)
        if not props:
            raise ValueError("Tidak ada region")

        # Ambil objek terbesar
        p = max(props, key=lambda x: x.area)
        return {
            "area": p.area,
            "perimeter": p.perimeter,
            "solidity": p.solidity,
            "eccentricity": p.eccentricity,
        }
    except Exception:
        return {k: 0 for k in ["area", "perimeter", "solidity", "eccentricity"]}


# === [3] PROSES UTAMA ===
try:
    df = pd.read_csv(CSV_METADATA_INPUT)
except FileNotFoundError:
    print(f"FATAL: File {CSV_METADATA_INPUT} tidak ditemukan.")
    exit()

all_features = []
print(f"Memulai ekstraksi fitur untuk {len(df)} pasangan gambar + masker...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    img_path = row["path_img"]
    mask_path = row["path_mask"]
    kelas = row["kelas"]
    split = row["split"]
    nama_file = row["nama_file_baru"]

    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        print(f"[SKIP] File hilang: {nama_file}")
        continue

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print(f"[SKIP] Gagal baca: {nama_file}")
        continue

    color_feats = extract_color_features(img)
    texture_feats = extract_texture_features(img)
    shape_feats = extract_shape_features(mask)

    all_features.append({
        "nama_file": nama_file,
        **color_feats,
        **texture_feats,
        **shape_feats,
        "kelas": kelas,
        "split": split
    })

# === [4] SIMPAN CSV ===
df_feat = pd.DataFrame(all_features)
df_feat.to_csv(CSV_OUTPUT_FITUR, index=False, encoding="utf-8-sig")

print("\n✅ Ekstraksi Fitur v16 Selesai!")
print(f"Total data: {len(df_feat)}")
print(f"Hasil disimpan di: {CSV_OUTPUT_FITUR}")
print("Lanjutkan ke training_v14.py atau model CNN/tabular berikutnya.")

# === [5] PLOT BERDASARKAN KELAS + LABEL NAMA KELAS ===
import matplotlib.pyplot as plt

print("\nMembuat grafik fitur berdasarkan kelas...")

# Urutkan dataframe berdasarkan kelas
df_plot = df_feat.sort_values(by="kelas").reset_index(drop=True)

# Ambil kelas unik dalam urutan kemunculan
kelas_unique = list(df_plot["kelas"].unique())

# Cari batas akhir setiap kelas
boundary_positions = []
for k in kelas_unique:
    last_index = df_plot[df_plot["kelas"] == k].index[-1]
    boundary_positions.append(last_index)

# Hitung posisi tengah setiap kelas untuk text label
mid_positions = []
start = 0
for end in boundary_positions:
    mid_positions.append((start + end) / 2)
    start = end + 1

# =====================================
# GRAFIK 1 — CONTRAST & ENERGY + LABEL KELAS
# =====================================
plt.figure(figsize=(14, 6))

plt.plot(df_plot.index, df_plot["energy"], 'o-', markersize=3, label="Energy", alpha=0.8)
plt.plot(df_plot.index, df_plot["contrast"], 'o-', markersize=3, label="Contrast", alpha=0.8)

# Garis batas antar kelas
for x in boundary_positions:
    plt.axvline(x=x, color='blue', linestyle='--', linewidth=1)

# Tampilkan nama kelas di atas segmen
for pos, nama_kelas in zip(mid_positions, kelas_unique):
    plt.text(pos, max(df_plot["contrast"].max(), df_plot["energy"].max()) + 0.05,
             nama_kelas,
             ha='center', va='bottom',
             fontsize=10, fontweight='bold')

plt.title("Fitur Contrast dan Energy")
plt.xlabel("Data (diurutkan berdasarkan kelas)")
plt.ylabel("Nilai Fitur")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# =====================================
# GRAFIK 2 — HSV + LABEL KELAS
# =====================================
plt.figure(figsize=(14, 6))

plt.plot(df_plot.index, df_plot["h_mean"], label="Hue Mean")
plt.plot(df_plot.index, df_plot["s_mean"], label="Saturation Mean")
plt.plot(df_plot.index, df_plot["v_mean"], label="Value Mean")

# Garis batas antar kelas
for x in boundary_positions:
    plt.axvline(x=x, color='red', linestyle='--', linewidth=1)

# Label kelas
max_hsv = max(df_plot["h_mean"].max(), df_plot["s_mean"].max(), df_plot["v_mean"].max())
for pos, nama_kelas in zip(mid_positions, kelas_unique):
    plt.text(pos, max_hsv + 2,
             nama_kelas,
             ha='center', va='bottom',
             fontsize=10, fontweight='bold')

plt.title("Fitur Warna HSV")
plt.xlabel("Data (diurutkan berdasarkan kelas)")
plt.ylabel("Nilai HSV")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Grafik selesai dibuat.")
