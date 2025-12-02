import cv2
import numpy as np
import pandas as pd
import os
import shutil
from tqdm import tqdm # Untuk progress bar

# --- [LIBRARY BARU] ---
try:
    from skimage.morphology import remove_small_objects # Ini adalah 'bwareaopen'
    from scipy.ndimage import binary_fill_holes       # Ini adalah 'Hole Filling'
except ImportError:
    print("FATAL ERROR: Library 'scikit-image' atau 'scipy' belum ter-install.")
    print("Silakan jalankan: pip install scikit-image scipy")
    exit()

print("--- Memulai Pipeline Penelitian ---")

# --- [1] KONFIGURASI PATH ---
BASE_PATH = r"D:\TEKOM THINGS\SEMESTER 5\PCD\TUGAS AKHIR"

# INPUT: CSV metadata mentah ANDA
CSV_INPUT = os.path.join(BASE_PATH, "Metadata_Jagung.xlsx") 

# Folder output (sama seperti sebelumnya)
OUTPUT_FOLDER_PIPELINE = os.path.join(BASE_PATH, "HASIL_PIPELINE_LENGKAP")
OUTPUT_FOLDER_1_PREPRO = os.path.join(OUTPUT_FOLDER_PIPELINE, "1_Hasil_Preprocessing")
OUTPUT_FOLDER_2_MASK_KOTOR = os.path.join(OUTPUT_FOLDER_PIPELINE, "2_Masker_Otsu_Kotor")
OUTPUT_FOLDER_3_MASK_BWAREAOPEN = os.path.join(OUTPUT_FOLDER_PIPELINE, "3_Masker_Morfologi_Bwareaopen")
OUTPUT_FOLDER_4_MASK_HOLE_FILLING = os.path.join(OUTPUT_FOLDER_PIPELINE, "4_Masker_Morfologi_HoleFilling")
OUTPUT_FOLDER_5_FINAL = os.path.join(OUTPUT_FOLDER_PIPELINE, "5_Hasil_Final_Morfologi")
CSV_OUTPUT = os.path.join(BASE_PATH, "dataset_hasil_pipeline.csv")

# Hapus folder lama jika ada agar bersih
if os.path.exists(OUTPUT_FOLDER_PIPELINE):
    shutil.rmtree(OUTPUT_FOLDER_PIPELINE)

# Buat semua folder baru
os.makedirs(OUTPUT_FOLDER_1_PREPRO)
os.makedirs(OUTPUT_FOLDER_2_MASK_KOTOR)
os.makedirs(OUTPUT_FOLDER_3_MASK_BWAREAOPEN)
os.makedirs(OUTPUT_FOLDER_4_MASK_HOLE_FILLING)
os.makedirs(OUTPUT_FOLDER_5_FINAL)
print(f"Folder output baru dibuat di: {OUTPUT_FOLDER_PIPELINE}")

# --- [PARAMETER] ---
TARGET_SIZE = 800

# --- [2] FUNGSI-FUNGSI ALUR KERJA (Tidak berubah) ---

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

# --- [3] PROSES BATCH UTAMA (MODIFIKASI v6) ---

try:
    if CSV_INPUT.endswith('.xlsx'):
        df = pd.read_excel(CSV_INPUT)
    else:
        df = pd.read_csv(CSV_INPUT) 
    
    # --- [MODIFIKASI 1] ---
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
    catatan = row['catatan'] # <-- DIAMBIL DARI CSV
    
    # Tentukan SEMUA path output baru
    path_prepro = os.path.join(OUTPUT_FOLDER_1_PREPRO, fname)
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
        hasil_segementasi = langkah_2_segmentasi_otsu(img_hsv)
        mask_bwareaopen, mask_no_bintik_bool = langkah_3a_bwareaopen(hasil_segementasi)
        mask_hole_filling = langkah_3b_hole_filling(mask_no_bintik_bool)
        img_final = langkah_4_isolasi_objek(img_preprocessed, mask_hole_filling)
        
        # === Simpan SEMUA hasil perantara ===
        cv2.imwrite(path_prepro, img_preprocessed)
        cv2.imwrite(path_segmentasi_otsu, hasil_segementasi)
        cv2.imwrite(path_mask_bwareaopen, mask_bwareaopen)
        cv2.imwrite(path_mask_hole_filling, mask_hole_filling)
        cv2.imwrite(path_final, img_final)
        
        # --- [MODIFIKASI 2] ---
        # Catat SEMUA path ke CSV, dan BAWA SERTA 'kelas', 'split', dan 'catatan'
        new_records.append({
            "nama_file": fname,
            "path_asli": input_path,
            "path_prepro": path_prepro,
            "path_segmentasi_otsu": path_segmentasi_otsu,
            "path_mask_bwareaopen": path_mask_bwareaopen,
            "path_mask_hole_filling": path_mask_hole_filling,
            "path_final_morfologi": path_final,
            "kelas": kelas,  
            "split": split,
            "catatan": catatan  # <- DITAMBAHKAN
        })
            
    except Exception as e:
        print(f"[ERROR] Gagal memproses {fname}: {e}")

# --- [4] SIMPAN CSV HASIL AKHIR ---
df_out = pd.DataFrame(new_records)
df_out.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")

print("\n=== Pipeline Selesai ===")
print(f"Semua hasil tahapan disimpan di: {OUTPUT_FOLDER_PIPELINE}")
print(f"CSV metadata baru disimpan di: {CSV_OUTPUT}")
print("\n--- Langkah Anda Selanjutnya ---")
print("1. Cek 'dataset_hasil_pipeline.csv'. Kolom 'kelas', 'split', dan 'catatan' seharusnya sudah terisi otomatis.")
print("2. (JIKA 'split' KOSONG) Harap isi kolom 'split' di 'dataset_raw.csv' Anda, lalu jalankan ulang skrip ini.")

print("3. Jalankan skrip 'augmentasi.py'. Skrip itu sekarang akan bekerja TANPA perlu edit manual.")
