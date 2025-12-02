import cv2
import numpy as np
import pandas as pd
import os
import shutil
from tqdm import tqdm # Untuk progress bar

# --- [LIBRARY BARU] ---
try:
    from skimage.morphology import remove_small_objects # Ini adalah 'bwareaopen'
    from scipy.ndimage import binary_fill_holes      # Ini adalah 'Hole Filling'
except ImportError:
    print("FATAL ERROR: Library 'scikit-image' atau 'scipy' belum ter-install.")
    print("Silakan jalankan: pip install scikit-image scipy")
    exit()

print("--- Memulai Pipeline Penelitian ---")

# --- [1] KONFIGURASI PATH (MODIFIKASI) ---
BASE_PATH = r"D:\TEKOM THINGS\SEMESTER 5\PCD\TUGAS AKHIR"

# INPUT: CSV metadata mentah ANDA
CSV_INPUT = os.path.join(BASE_PATH, "Metadata_Jagung.xlsx") 

# Folder output (sama seperti sebelumnya)
OUTPUT_FOLDER_PIPELINE = os.path.join(BASE_PATH, "2_HASIL_PIPELINE_LENGKAP")
OUTPUT_FOLDER_1_PREPRO = os.path.join(OUTPUT_FOLDER_PIPELINE, "1_Hasil_Preprocessing_BGR")

# --- [DEKLARASI FOLDER BARU (LEBIH SPESIFIK)] ---
OUTPUT_FOLDER_1a_HSV_H = os.path.join(OUTPUT_FOLDER_PIPELINE, "1a_Hasil_Warna_HSV", "H_Hue")
OUTPUT_FOLDER_1a_HSV_S = os.path.join(OUTPUT_FOLDER_PIPELINE, "1a_Hasil_Warna_HSV", "S_Saturation")
OUTPUT_FOLDER_1a_HSV_V = os.path.join(OUTPUT_FOLDER_PIPELINE, "1a_Hasil_Warna_HSV", "V_Value")

OUTPUT_FOLDER_1b_RGB_R = os.path.join(OUTPUT_FOLDER_PIPELINE, "1b_Hasil_Warna_RGB", "R_Red")
OUTPUT_FOLDER_1b_RGB_G = os.path.join(OUTPUT_FOLDER_PIPELINE, "1b_Hasil_Warna_RGB", "G_Green")
OUTPUT_FOLDER_1b_RGB_B = os.path.join(OUTPUT_FOLDER_PIPELINE, "1b_Hasil_Warna_RGB", "B_Blue")
# --------------------------------------------------

OUTPUT_FOLDER_2_MASK_KOTOR = os.path.join(OUTPUT_FOLDER_PIPELINE, "2_Masker_Otsu_Kotor")
OUTPUT_FOLDER_3_MASK_BWAREAOPEN = os.path.join(OUTPUT_FOLDER_PIPELINE, "3_Masker_Morfologi_Bwareaopen")
OUTPUT_FOLDER_4_MASK_HOLE_FILLING = os.path.join(OUTPUT_FOLDER_PIPELINE, "4_Masker_Morfologi_HoleFilling")
OUTPUT_FOLDER_5_FINAL = os.path.join(OUTPUT_FOLDER_PIPELINE, "5_Hasil_Final_Morfologi")
CSV_OUTPUT = os.path.join(BASE_PATH, "2_dataset_hasil_pipeline.csv")

# Hapus folder lama jika ada agar bersih
if os.path.exists(OUTPUT_FOLDER_PIPELINE):
    shutil.rmtree(OUTPUT_FOLDER_PIPELINE)

# Buat semua folder baru
os.makedirs(OUTPUT_FOLDER_1_PREPRO)

# --- [PEMBUATAN FOLDER BARU DENGAN SUBFOLDER] ---
os.makedirs(OUTPUT_FOLDER_1a_HSV_H)
os.makedirs(OUTPUT_FOLDER_1a_HSV_S)
os.makedirs(OUTPUT_FOLDER_1a_HSV_V)

os.makedirs(OUTPUT_FOLDER_1b_RGB_R)
os.makedirs(OUTPUT_FOLDER_1b_RGB_G)
os.makedirs(OUTPUT_FOLDER_1b_RGB_B)
# -------------------------------------------------

os.makedirs(OUTPUT_FOLDER_2_MASK_KOTOR)
os.makedirs(OUTPUT_FOLDER_3_MASK_BWAREAOPEN)
os.makedirs(OUTPUT_FOLDER_4_MASK_HOLE_FILLING)
os.makedirs(OUTPUT_FOLDER_5_FINAL)
print(f"Folder output baru dibuat di: {OUTPUT_FOLDER_PIPELINE}")

# --- [PARAMETER] ---
TARGET_SIZE = 800

# --- [2] FUNGSI-FUNGSI ALUR KERJA (Dimodifikasi) ---

def apply_gamma_correction(img, gamma_val=0.7):
    invGamma = 1.0 / gamma_val
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def langkah_1_preprocessing_lengkap(img, target_size=TARGET_SIZE):
    h, w = img.shape[:2]
    if h > w: new_h, new_w = target_size, int((w / h) * target_size)
    else: new_h, new_w = int((h / w) * target_size), target_size
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized
    img_brightened = apply_gamma_correction(canvas) 
    img_hsv = cv2.cvtColor(img_brightened, cv2.COLOR_BGR2HSV)
    return img_brightened, img_hsv

# --- [FUNGSI BARU DIMODIFIKASI] ---
def langkah_1_1_simpan_konversi_warna(img_brightened, img_hsv, fname):
    
    # Simpan RGB (Sebagai BGR pre-processed, lalu pisahkan channel R, G, B)
    B, G, R = cv2.split(img_brightened) 
    path_R = os.path.join(OUTPUT_FOLDER_1b_RGB_R, fname)
    path_G = os.path.join(OUTPUT_FOLDER_1b_RGB_G, fname)
    path_B = os.path.join(OUTPUT_FOLDER_1b_RGB_B, fname)
    cv2.imwrite(path_R, R)
    cv2.imwrite(path_G, G)
    cv2.imwrite(path_B, B)

    # Simpan HSV (Pisahkan channel H, S, V)
    H, S, V = cv2.split(img_hsv) 
    path_H = os.path.join(OUTPUT_FOLDER_1a_HSV_H, fname)
    path_S = os.path.join(OUTPUT_FOLDER_1a_HSV_S, fname)
    path_V = os.path.join(OUTPUT_FOLDER_1a_HSV_V, fname)
    cv2.imwrite(path_H, H)
    cv2.imwrite(path_S, S)
    cv2.imwrite(path_V, V)

    # Return SEMUA path yang sudah di-save
    return path_R, path_G, path_B, path_H, path_S, path_V
# ------------------------------------

def langkah_2_segmentasi_otsu(img_hsv):
    v_channel = img_hsv[:, :, 2]
    ret, mask_otsu = cv2.threshold(v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask_otsu

def langkah_3a_bwareaopen(hasil_segementasi):
    mask_bool = hasil_segementasi > 0 
    min_size_noise = 150 
    mask_no_bintik_bool = remove_small_objects(mask_bool, min_size=min_size_noise)
    mask_no_bintik_img = (mask_no_bintik_bool * 255).astype(np.uint8)
    return mask_no_bintik_img, mask_no_bintik_bool

def langkah_3b_hole_filling(mask_no_bintik_bool):
    mask_filled_bool = binary_fill_holes(mask_no_bintik_bool)
    mask_filled_img = (mask_filled_bool * 255).astype(np.uint8)
    return mask_filled_img

def langkah_4_isolasi_objek(img_brightened, mask_hole_filling):
    return cv2.bitwise_and(img_brightened, img_brightened, mask=mask_hole_filling)

# --- [3] PROSES BATCH UTAMA (MODIFIKASI) ---

try:
    if CSV_INPUT.endswith('.xlsx'):
        df = pd.read_excel(CSV_INPUT)
    else:
        df = pd.read_csv(CSV_INPUT) 
    
    # Cek semua kolom yang kita butuhkan
    required_cols = ['nama_file', 'path_asli', 'kelas', 'catatan']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Kolom wajib '{col}' tidak ditemukan di {CSV_INPUT}.")
    
    # Cek kolom 'split' (opsional tapi disarankan)
    if 'split' not in df.columns: 
        print("\n--- PERINGATAN PENTING ---")
        print("Kolom 'split' (Latih/Validasi/Uji) tidak ditemukan di 'dataset_raw.csv'.")
        print("Anda HARUS menambahkannya secara manual ke 'dataset_raw.csv' sebelum bisa menjalankan skrip 'augmentasi.py'.")
        df['split'] = 'BELUM_DIISI' 
    
    print(f"Membaca {len(df)} file dari {CSV_INPUT}")

except FileNotFoundError:
    print(f"FATAL: File {CSV_INPUT} ('dataset_raw.csv') tidak ditemukan. Skrip tidak bisa lanjut.")
    exit()
except ValueError as e:
    print(f"FATAL: Error di CSV: {e}")
    exit()
except Exception as e:
    print(f"FATAL: Gagal membaca CSV ({e}).")
    exit()


new_records = []
print(f"Memulai pipeline untuk {len(df)} gambar...")

for i, row in tqdm(df.iterrows(), desc="Memproses Gambar"):
    
    # Ambil data dari CSV mentah
    fname = row['nama_file']
    input_path = row['path_asli'] 
    kelas = row['kelas']
    split = row['split']
    catatan = row['catatan'] 
    
    # Tentukan SEMUA path output baru
    path_prepro = os.path.join(OUTPUT_FOLDER_1_PREPRO, fname)
    # Path untuk channel warna akan di-return oleh fungsi langkah_1_1_simpan_konversi_warna
    path_segmentasi_otsu = os.path.join(OUTPUT_FOLDER_2_MASK_KOTOR, fname)
    path_mask_bwareaopen = os.path.join(OUTPUT_FOLDER_3_MASK_BWAREAOPEN, fname)
    path_mask_hole_filling = os.path.join(OUTPUT_FOLDER_4_MASK_HOLE_FILLING, fname)
    path_final = os.path.join(OUTPUT_FOLDER_5_FINAL, fname)

    if not os.path.exists(input_path):
        print(f"[SKIP] path_asli tidak ditemukan: {input_path}")
        continue
        
    img_raw = cv2.imread(input_path)
    if img_raw is None: 
        print(f"[SKIP] Gagal membaca gambar: {input_path}")
        continue

    try:
        # === ALUR KERJA LINEAR YANG RAPI ===
        img_preprocessed, img_hsv = langkah_1_preprocessing_lengkap(img_raw)
        
        # --- [LANGKAH BARU: Simpan Hasil Konversi Warna ke Subfolder] ---
        path_R, path_G, path_B, path_H, path_S, path_V = langkah_1_1_simpan_konversi_warna(
            img_preprocessed, 
            img_hsv, 
            fname
        )
        # ----------------------------------------------------
        
        hasil_segementasi = langkah_2_segmentasi_otsu(img_hsv)
        mask_bwareaopen, mask_no_bintik_bool = langkah_3a_bwareaopen(hasil_segementasi)
        mask_hole_filling = langkah_3b_hole_filling(mask_no_bintik_bool)
        img_final = langkah_4_isolasi_objek(img_preprocessed, mask_hole_filling)
        
        # === Simpan SEMUA hasil perantara (yang lama) ===
        cv2.imwrite(path_prepro, img_preprocessed)
        cv2.imwrite(path_segmentasi_otsu, hasil_segementasi)
        cv2.imwrite(path_mask_bwareaopen, mask_bwareaopen)
        cv2.imwrite(path_mask_hole_filling, mask_hole_filling)
        cv2.imwrite(path_final, img_final)
        
        # --- [Pencatatan Path ke CSV] ---
        new_records.append({
            "nama_file": fname,
            "path_asli": input_path,
            "path_prepro_bgr": path_prepro,
            # --- [PENAMBAHAN KE CSV (Path sekarang mengarah ke subfolder)] ---
            "path_channel_R": path_R,
            "path_channel_G": path_G,
            "path_channel_B": path_B,
            "path_channel_H": path_H,
            "path_channel_S": path_S,
            "path_channel_V": path_V,
            # -----------------------------------------------------------------
            "path_segmentasi_otsu": path_segmentasi_otsu,
            "path_mask_bwareaopen": path_mask_bwareaopen,
            "path_mask_hole_filling": path_mask_hole_filling,
            "path_final_morfologi": path_final,
            "kelas": kelas,  
            "split": split,
            "catatan": catatan 
        })
            
    except Exception as e:
        print(f"[ERROR] Gagal memproses {fname}: {e}")

# --- [4] SIMPAN CSV HASIL AKHIR ---
df_out = pd.DataFrame(new_records)
df_out.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")

print("\n=== Pipeline Selesai ===")
print(f"Semua hasil tahapan disimpan di: {OUTPUT_FOLDER_PIPELINE}")
print(f"Channel warna kini tersimpan dalam subfolder di: {os.path.join(OUTPUT_FOLDER_PIPELINE, '1a_Hasil_Warna_HSV')} dan {os.path.join(OUTPUT_FOLDER_PIPELINE, '1b_Hasil_Warna_RGB')}")
print(f"CSV metadata baru disimpan di: {CSV_OUTPUT}")
print("\n--- Langkah Anda Selanjutnya ---")
print("1. Cek struktur folder dan 'dataset_hasil_pipeline.csv' untuk memastikan path sudah benar.")
print("2. (JIKA 'split' KOSONG) Harap isi kolom 'split' di 'dataset_raw.csv' Anda, lalu jalankan ulang skrip ini.")
print("3. Jalankan skrip 'augmentasi.py'.")
