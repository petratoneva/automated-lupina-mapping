# ── config_s2.py ──────────────────────────────────────────────────────────────
# Sentinel-2 lupine mapping pipeline configuration
# Import in all S2 notebooks: from config_s2 import *

import os
import multiprocessing

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = r"C:\Users\petra\Documents\NATT_Test"
DATA_DIR  = os.path.join(BASE_DIR, "data")
PRED_DIR  = os.path.join(BASE_DIR, "predictions")
MODEL_DIR = os.path.join(BASE_DIR, "models")
TMP_DIR   = os.path.join(BASE_DIR, "tmp_s2")

# S2 specific data paths
S2_DIR      = os.path.join(DATA_DIR, "sentinel2")
S2_IMG_DIR  = os.path.join(S2_DIR, "images")
S2_MASK_DIR = os.path.join(S2_DIR, "masks")
S2_GRID_PATH = os.path.join(DATA_DIR, "s2_grid.gpkg")

# Shared with aerial pipeline
REGION_PATH     = os.path.join(DATA_DIR, "regions.gpkg")
LUPINE_VEC      = os.path.join(DATA_DIR, "lupine.gpkg")
LUPINE_LAYER    = "L14.4 Alaskalúpína"
CULTIVATED_PATH = os.path.join(DATA_DIR, "cultivated_land_2018-2020.gpkg")

# ── Mode ──────────────────────────────────────────────────────────────────────
MODE          = "all"   # 'region' or 'all'
REGION_ID     = 21
REGION_ID_COL = "id"

# ── S2 imagery settings ───────────────────────────────────────────────────────
S2_BANDS   = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S2_RES     = 10.0      # resolution in metres
S2_SCALE   = 10000.0   # DN = reflectance × 10000
NODATA_VAL = 32767     # int16 max = nodata

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME_S2 = "rf_s2"
MODEL_PATH_S2 = os.path.join(MODEL_DIR, "sentinel2", f"{MODEL_NAME_S2}_model.joblib")

# ── Training ──────────────────────────────────────────────────────────────────
N_PIXELS_PER_TILE = 5000
RANDOM_SEED       = 42
TRAIN_SPLIT       = 0.8

# ── Prediction ────────────────────────────────────────────────────────────────
PIXEL_THRESHOLDS  = [0.5]
DELETE_AFTER      = True
CHECKPOINT_EVERY  = 50

# ── Postprocessing ────────────────────────────────────────────────────────────
CULT_BUFFER_M = 20  # metres — buffer around cultivated polygons


# ── Output dirs ───────────────────────────────────────────────────────────────
def s2_model_out_name():
    if MODE == "region":
        return f"{MODEL_NAME_S2}_region{REGION_ID}"
    return MODEL_NAME_S2
S2_PRED_DIR   = os.path.join(PRED_DIR, "sentinel2")
S2_MODEL_OUT = os.path.join(S2_PRED_DIR, s2_model_out_name())
S2_PP_OUT_DIR = os.path.join(S2_MODEL_OUT, "postprocessed")

def s2_binary_tile_dir(threshold):
    return os.path.join(S2_MODEL_OUT, f"binary_tiles_t{threshold}")

def s2_prob_tile_dir():
    return os.path.join(S2_MODEL_OUT, "prob_tiles")

def s2_pp_tile_dir(threshold):
    return os.path.join(S2_MODEL_OUT, f"pp_tiles_t{threshold}")

def s2_vector_tile_dir(threshold):
    return os.path.join(S2_MODEL_OUT, f"vector_tiles_t{threshold}")

def s2_output_suffix():
    return f"region_{REGION_ID}" if MODE == "region" else "all_tiles"

# ── Create output directories ─────────────────────────────────────────────────
_dirs = (
    [S2_DIR, S2_IMG_DIR, S2_MASK_DIR, TMP_DIR,
     S2_PRED_DIR, S2_MODEL_OUT, S2_PP_OUT_DIR,
     os.path.join(MODEL_DIR, "sentinel2"),
     s2_prob_tile_dir()] +
    [s2_binary_tile_dir(t) for t in PIXEL_THRESHOLDS] +
    [s2_pp_tile_dir(t) for t in PIXEL_THRESHOLDS] +
    [s2_vector_tile_dir(t) for t in PIXEL_THRESHOLDS]
)

for d in _dirs:
    os.makedirs(d, exist_ok=True)

# ── System info ───────────────────────────────────────────────────────────────
try:
    import psutil
    RAM_GB = psutil.virtual_memory().total / 1e9
except ImportError:
    RAM_GB = None

N_CORES = multiprocessing.cpu_count()

# ── Download URLs for auto-fetch ──────────────────────────────────────────────
_DATA_URLS_S2 = {
    S2_GRID_PATH: (
        "https://ftp.natt.is/public/marco/download/sentinel-2-mosaic/"
        "s2_median_mosaic_2023-2025_10km_grid_crs8088.gpkg"
    ),
    CULTIVATED_PATH: (
        "http://gis.fasteignaskra.is/geoserver/public/ows?service=WFS&version=1.0.0"
        "&request=GetFeature&typeName=public%3ARaektad_land_2018_2020"
        "&outputFormat=application%2Fx-gpkg"
    ),
}

# ── File checks ───────────────────────────────────────────────────────────────
def check_required_files_s2():
    import requests

    checks = [
        (S2_GRID_PATH,    "S2 grid"),
        (REGION_PATH,     "Regions"),
        (LUPINE_VEC,      "Lupine polygons"),
        (CULTIVATED_PATH, "Cultivated land"),
    ]
    all_ok = True
    print("Checking required files:")
    for path, label in checks:
        if os.path.exists(path):
            print(f"  ✓ {label}: {os.path.basename(path)}")
            continue

        if path in _DATA_URLS_S2:
            print(f"  [!] Missing: {label} — downloading …")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            try:
                r = requests.get(_DATA_URLS_S2[path], stream=True, timeout=120)
                if r.status_code == 200:
                    with open(path, "wb") as f:
                        for chunk in r.iter_content(1024*1024):
                            f.write(chunk)
                    print(f"      ✓ Downloaded → {os.path.basename(path)}")
                else:
                    print(f"      [!] HTTP {r.status_code} — download failed")
                    all_ok = False
            except Exception as e:
                print(f"      [!] Download failed: {e}")
                all_ok = False
        else:
            print(f"  [!] MISSING  {label}: {os.path.basename(path)}")
            all_ok = False

    if not all_ok:
        raise FileNotFoundError("Required files missing — check paths in config_s2.py")
    print("All files found ✓")

if __name__ == "__main__":
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"Cores: {N_CORES}  |  RAM: {RAM_GB:.0f} GB" if RAM_GB else f"Cores: {N_CORES}")
    check_required_files_s2()