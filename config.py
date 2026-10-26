# ── config.py ─────────────────────────────────────────────────────────────────
# Central configuration for the lupine mapping pipeline.
# Import in all notebooks: from config import *

import os
import multiprocessing

# ── 1. Core paths ─────────────────────────────────────────────────────────────
BASE_DIR  = r"C:\Users\petra\Documents\NATT_Test"
DATA_DIR  = os.path.join(BASE_DIR, "data")
PRED_DIR  = os.path.join(BASE_DIR, "predictions")
MODEL_DIR = os.path.join(BASE_DIR, "models")
TMP_DIR   = os.path.join(BASE_DIR, "tmp")

# ── 2. Data paths ─────────────────────────────────────────────────────────────
GRID_PATH       = os.path.join(DATA_DIR, "grunnkort_loftmynd_nidurhalgeom.shp")
LUPINE_VEC      = os.path.join(DATA_DIR, "lupine.gpkg")
LUPINE_LAYER    = "L14.4 Alaskalúpína"
CULTIVATED_PATH = os.path.join(DATA_DIR, "cultivated_land_2018-2020.gpkg")
REGION_PATH     = os.path.join(DATA_DIR, "regions.gpkg")

# ── 3. Mode ───────────────────────────────────────────────────────────────────
# 'region' = tiles overlapping a region polygon
# 'all'    = every tile in the grid
MODE          = "region"
REGION_ID     = 3
REGION_ID_COL = "id"

# ── 4. Prepare paths (010) ────────────────────────────────────────────────────
DOWNLOADED_DIR = os.path.join(DATA_DIR, "downloaded_tiles")
MASKS_DIR      = os.path.join(DATA_DIR, "masks")
TILES_DIR      = os.path.join(DATA_DIR, "tiles")
TILES_IMG_DIR  = os.path.join(TILES_DIR, "images")
TILES_MASK_DIR = os.path.join(TILES_DIR, "masks")
QA_VISUALS_DIR = os.path.join(BASE_DIR, "qa_visuals")
MASK_QA_PATH   = os.path.join(DATA_DIR, "mask_qa_full.csv")
SUBSET_OUT     = os.path.join(DATA_DIR, "subset_500_random_seed99.csv")

# ── 5. Training settings (020) ────────────────────────────────────────────────
MODEL_TYPE       = "rf"    # "rf" or "xgb"
USE_FILTER       = False   # True = NDVI purity filter (0.25–0.50), False = no filter
USE_NIR_CONTRAST = False   # True = include NIR_contrast_25 texture feature
USE_SUBSET       = False   # True = train on 500-tile subset, False = all tiles
SUBSET_SIZE      = 500     # only used if USE_SUBSET = True

SAMPLES_PER_TILE = 5000
NDVI_MIN         = 0.25   # purity filter lower bound
NDVI_MAX         = 0.50   # purity filter upper bound
RANDOM_STATE     = 42
TEST_SIZE        = 0.2
RF_N_ESTIMATORS  = 50
RF_MAX_DEPTH     = 15
XGB_N_ESTIMATORS = 50
XGB_MAX_DEPTH    = 15
XGB_LR           = 0.1

# ── 6. Model (update after training) ─────────────────────────────────────────
MODEL_NAME = "v11_rf_nofilter_notexture_all"   # select model to use for 101-105 scripts
MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_model.joblib")

# ── 7. Prediction settings ────────────────────────────────────────────────────
PIXEL_THRESHOLDS     = [0.5, 0.7]
DELETE_AFTER_PREDICT = True
CHECKPOINT_EVERY     = 50

# ── 8. Postprocessing settings ────────────────────────────────────────────────
MEDIAN_SIZE       = 3      # salt/pepper removal (px). 0 = skip
OPENING_SIZE      = 4      # thin FP removal (px). 0 = skip
CLOSING_SIZE      = 5      # gap filling (px). 0 = skip
BRIGHTNESS_THRESH = 0.2    # shadow mask brightness threshold
APPLY_SHADOW_MASK = True   # False = skip shadow masking in 103
DISSOLVE_BUFFER_M = 1      # bridges thin gaps at tile edges

# ── 9. MMU filter ─────────────────────────────────────────────────────────────
MMU_SIZES = [(1, "1m"), (4, "2m"), (25, "5m")]  # (min_m2, label)

# ── 10. Output dir functions ──────────────────────────────────────────────────
def model_out_name():
    if MODE == "region":
        return f"{MODEL_NAME}_region{REGION_ID}"
    return MODEL_NAME

def output_suffix():
    return f"region_{REGION_ID}" if MODE == "region" else "all_tiles"

def shadow_suffix():
    return "_shadow" if APPLY_SHADOW_MASK else "_noshadow"

MODEL_OUT_DIR = os.path.join(PRED_DIR, model_out_name())
PROB_TILE_DIR = os.path.join(MODEL_OUT_DIR, "prob_tiles")
PP_OUT_DIR    = os.path.join(MODEL_OUT_DIR, "postprocessed")

def binary_tile_dir(threshold):
    return os.path.join(MODEL_OUT_DIR, f"binary_tiles_t{threshold}")

def pp_tile_dir(threshold):
    return os.path.join(MODEL_OUT_DIR, f"pp_tiles_t{threshold}{shadow_suffix()}")

def shadow_tile_dir():
    return os.path.join(MODEL_OUT_DIR, "shadow_tiles")

def vector_tile_dir(threshold, min_label):
    return os.path.join(MODEL_OUT_DIR, f"vector_tiles_t{threshold}_min{min_label}{shadow_suffix()}")

# ── 11. Train paths (020) ─────────────────────────────────────────────────────
SUBSET_PATH      = os.path.join(DATA_DIR, "subset_500_random_seed99.csv")
CACHE_FILTERED   = os.path.join(DATA_DIR, "pixel_indices_filtered{}.json".format(
                       "_subset" if USE_SUBSET else "_all"))
CACHE_UNFILTERED = os.path.join(DATA_DIR, "pixel_indices_unfiltered{}.json".format(
                       "_subset" if USE_SUBSET else "_all"))
RESULTS_PATH     = os.path.join(MODEL_DIR, "model_comparison.csv")
FIXED_TILES_PATH = os.path.join(DATA_DIR, "qa_fixed_tiles.json")

# ── 12. System info ───────────────────────────────────────────────────────────
try:
    import psutil
    RAM_GB = psutil.virtual_memory().total / 1e9
except ImportError:
    RAM_GB = None

N_CORES = multiprocessing.cpu_count()

# ── 13. Create directories ────────────────────────────────────────────────────
_dirs = (
    [MODEL_OUT_DIR, PROB_TILE_DIR, PP_OUT_DIR, TMP_DIR, shadow_tile_dir(),
     DOWNLOADED_DIR, MASKS_DIR, TILES_IMG_DIR, TILES_MASK_DIR,
     QA_VISUALS_DIR, MODEL_DIR] +
    [binary_tile_dir(t) for t in PIXEL_THRESHOLDS] +
    [pp_tile_dir(t) for t in PIXEL_THRESHOLDS] +
    [vector_tile_dir(t, label) for t in PIXEL_THRESHOLDS for _, label in MMU_SIZES]
)
for d in _dirs:
    os.makedirs(d, exist_ok=True)

# ── 14. Auto-download URLs + file checks ──────────────────────────────────────
_DATA_URLS = {
    GRID_PATH: (
        "https://ogc.gis.is/geoserver/loftmyndir/ows?service=WFS&version=1.0.0"
        "&request=GetFeature&typeName=loftmyndir%3Agrunnkort_loftmynd_nidurhal"
        "&outputFormat=application%2Fx-gpkg&maxFeatures=100000"
    ),
    CULTIVATED_PATH: (
        "http://gis.fasteignaskra.is/geoserver/public/ows?service=WFS&version=1.0.0"
        "&request=GetFeature&typeName=public%3ARaektad_land_2018_2020"
        "&outputFormat=application%2Fx-gpkg"
    ),
}

def check_required_files():
    import requests

    checks = [
        (GRID_PATH,       "Grid shapefile"),
        (MODEL_PATH,      "Model file"),
        (CULTIVATED_PATH, "Cultivated land"),
    ]
    if MODE == "region":
        checks.append((REGION_PATH, "Regions GeoPackage"))

    all_ok = True
    print("Checking required files:")
    for path, label in checks:
        if os.path.exists(path):
            print(f"  ✓ {label}: {os.path.basename(path)}")
            continue

        if path in _DATA_URLS:
            print(f"  [!] Missing: {label} — downloading …")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            try:
                r = requests.get(_DATA_URLS[path], stream=True, timeout=120)
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
        raise FileNotFoundError("Required files missing — check paths in config.py")
    print("All files found ✓")

# ── 15. __main__ ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"Cores: {N_CORES}  |  RAM: {RAM_GB:.0f} GB" if RAM_GB else f"Cores: {N_CORES}")
    check_required_files()
