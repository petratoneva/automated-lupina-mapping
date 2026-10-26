# Automated Lupine Mapping Pipeline

Automated detection and mapping of *Lupinus nootkatensis* (Alaskan lupine) across Iceland using aerial imagery and Sentinel-2 satellite data. Developed at Náttúrufræðistofnun Íslands (Natural Science Institute of Iceland) as part of a national lupine distribution mapping project.

---

## Overview

The pipeline consists of two independent but complementary mapping approaches:

| | Aerial | Sentinel-2 |
|---|---|---|
| Resolution | 0.25 m | 10 m |
| Coverage | Eastfjords + South Iceland | All Iceland |
| Training area | Eastfjords | Eastfjords |
| Features | 7–9 spectral + texture | 15 spectral |
| Model | Random Forest / XGBoost | Random Forest |
| Output | High-detail polygons | Wall-to-wall coverage |

---

## Repository Structure

```
├── config.py                  # Aerial pipeline settings and paths
├── config_s2.py               # Sentinel-2 pipeline settings and paths
├── helpers.py                 # Shared helper functions (aerial)
├── helpers_s2.py              # Helper functions (Sentinel-2)
├── requirements.txt           # Python dependencies
│
├── 010_prepare_aerial.ipynb   # Download tiles, create masks, subtile to 512px
├── 020_train_aerial.ipynb     # Feature extraction and model training
├── 101_predict.ipynb          # Predict on all aerial tiles
├── 102_shadow.ipynb           # Generate shadow masks
├── 103_postprocess.ipynb      # Cultivated suppression, shadow, morphology
├── 104_vectorize.ipynb        # Vectorize postprocessed tiles
├── 105_mosaic_vectors.ipynb   # Merge tile polygons into final output
│
├── 200_satellite/
│   ├── 201_prepare_sat.ipynb  # Download S2 tiles, create masks
│   ├── 202_train_sat.ipynb    # Feature extraction and model training
│   ├── 203_predict_sat.ipynb  # Predict on all S2 tiles
│   ├── 204_postprocess_sat.ipynb  # Cultivated land suppression
│   └── 205_vectorize_sat.ipynb    # Vectorize and merge S2 predictions
```

---

## Setup

### Requirements

- Python 3.11
- Conda (recommended)

### Installation

```bash
conda create -n lupine python=3.11
conda activate lupine
pip install -r requirements.txt
pip install geoai-py fiona lxml  # aerial pipeline only
```

### Configuration

All paths and settings are controlled from `config.py` (aerial) and `config_s2.py` (Sentinel-2). The only required change before running is:

```python
# config.py
BASE_DIR = r"C:\path\to\your\project\folder"
```

Data files (aerial grid, cultivated land, S2 grid) are downloaded automatically on first run if not already present.

---

## Aerial Pipeline

### Step 1 — Prepare data (`010_prepare_aerial.ipynb`)

Downloads aerial tiles intersecting lupine reference polygons from the Náttúrufræðistofnun WFS endpoint. Rasterizes lupine reference polygons to create binary masks. Subtiles each 4,000 × 4,000 px source tile into 512 × 512 px training tiles, excluding tiles with no lupine pixels.

**Requires:** `geoai-py`, `fiona`, `lxml`
**Output:** `data/tiles/images/`, `data/tiles/masks/`

### Step 2 — Train model (`020_train_aerial.ipynb`)

Extracts spectral and texture features from training tiles, applies optional NDVI purity filter, and trains a Random Forest or XGBoost classifier.

**Key settings in `config.py`:**
```python
MODEL_TYPE       = "rf"      # "rf" or "xgb"
USE_FILTER       = True      # NDVI purity filter (0.25–0.50)
USE_NIR_CONTRAST = True      # Include NIR texture feature (25px window)
USE_SUBSET       = False     # False = train on all tiles
SAMPLES_PER_TILE = 5000      # Pixels sampled per tile
```

**Features (7–9):** R, G, B, NIR, NDVI, NDWI, EVI, NIR_contrast_5 (optional), NIR_contrast_25 (optional)

**Output:** `models/{model_name}_model.joblib`, `models/model_comparison.csv`

After training, update `MODEL_NAME` in `config.py` to match the saved model name before running prediction.

### Step 3 — Predict (`101_predict.ipynb`)

Downloads each aerial tile, runs the trained model, saves probability and binary prediction tiles at two thresholds (t=0.5 and t=0.7), and deletes the source tile to save disk space. Supports resume — already predicted tiles are skipped.

**Output:** `predictions/{model}/binary_tiles_t0.5/`, `predictions/{model}/prob_tiles/`

### Step 4 — Shadow masks (`102_shadow.ipynb`)

Generates per-tile shadow masks based on brightness threshold (mean of R, G, B, NIR < 0.2). Only processes tiles with at least one lupine pixel. Optional — can be skipped by setting `APPLY_SHADOW_MASK = False` in `config.py`.

**Output:** `predictions/{model}/shadow_tiles/`

### Step 5 — Postprocess (`103_postprocess.ipynb`)

Applies per-tile postprocessing in sequence:
1. Cultivated land suppression (national dataset 2018–2020)
2. Shadow masking (optional, controlled by `APPLY_SHADOW_MASK`)
3. Morphological cleanup: median filter (3px) → opening (4px) → closing (5px)

**Output:** `predictions/{model}/pp_tiles_t{thresh}/`

### Step 6 — Vectorize (`104_vectorize.ipynb`)

Converts postprocessed binary tiles to vector polygons. Applies minimum mapping unit (MMU) filter at three sizes (1m², 4m², 25m²). Supports resume.

**Output:** `predictions/{model}/vector_tiles_t{thresh}_min{label}/`

### Step 7 — Merge (`105_mosaic_vectors.ipynb`)

Merges all per-tile polygon files into a single output layer per threshold and MMU combination. Optionally dissolves patches at tile boundaries. Generates centroid points from the final polygon layer.

**Output:** `predictions/{model}/postprocessed/lupine_polygons_{suffix}_{model}_min{label}_t{thresh}_{shadow/noshadow}.gpkg`
---

## Sentinel-2 Pipeline

### Step 1 — Prepare data (`201_prepare_sat.ipynb`)

Downloads Sentinel-2 10km tiles from the Náttúrustofa cloudless median composite (Jul–Aug 2023–2025) that intersect lupine reference polygons. Rasterizes lupine masks to the 10m grid.

**Output:** `data/sentinel2/images/`, `data/sentinel2/masks/`

### Step 2 — Train model (`202_train_sat.ipynb`)

Extracts 15 spectral features from all pixels in training tiles (no subsampling, no NDVI filter). Trains a Random Forest classifier.

**Features (15):** B2, B3, B4, B8, B5, B8A, B11, NDVI, NDRE, CIre, EVI, NDWI, NDMI, SAVI, SWIR_ratio

**Output:** `models/sentinel2/rf_s2_model.joblib`

### Step 3 — Predict (`203_predict_sat.ipynb`)

Downloads each S2 tile, predicts lupine probability, saves binary tiles at t=0.5 and t=0.7. Supports resume. Set `MODE = "region"` and `REGION_ID` in `config_s2.py` to predict on a single region only.

**Output:** `predictions/sentinel2/{model}/binary_tiles_t{thresh}/`

### Step 4 — Postprocess (`204_postprocess_sat.ipynb`)

Applies cultivated land suppression with a 20m buffer around field boundaries. No morphological operations — at 10m resolution these would remove genuine lupine patches.

**Output:** `predictions/sentinel2/{model}/pp_tiles_t{thresh}/`

### Step 5 — Vectorize and merge (`205_vectorize_sat.ipynb`)

Vectorizes postprocessed S2 tiles with a minimum patch size of 1 pixel (100m²). Merges all tile polygons and generates centroids.

**Output:** `predictions/sentinel2/{model}/postprocessed/lupine_s2_polygons_{suffix}_{model}_t{thresh}.gpkg`

---

## Key Configuration Options

| Setting | File | Description |
|---|---|---|
| `BASE_DIR` | config.py | Root project directory |
| `MODEL_TYPE` | config.py | `"rf"` or `"xgb"` |
| `USE_FILTER` | config.py | NDVI purity filter during training |
| `USE_NIR_CONTRAST` | config.py | NIR texture feature (25px window) |
| `USE_SUBSET` | config.py | Train on 500-tile subset or all tiles |
| `APPLY_SHADOW_MASK` | config.py | Apply shadow masking in postprocessing |
| `PIXEL_THRESHOLDS` | config.py | Probability thresholds for binary output |
| `MMU_SIZES` | config.py | Minimum mapping unit sizes |
| `MODE` | config_s2.py | `"all"` or `"region"` |
| `REGION_ID` | config_s2.py | Region ID for regional prediction |

---

## Data Sources

- **Aerial imagery:** Náttúrufræðistofnun, *Frummyndir Grunnkort Loftmynd NÁTT*, 2025. 4-band GeoTIFF, ~25cm GSD. Available via WFS: `https://ogc.gis.is/geoserver/loftmyndir/ows`
- **Lupine reference polygons:** Náttúrufræðistofnun, layer *L14.4 Alaskalúpína*
- **Sentinel-2 composite:** Náttúrustofa cloudless median mosaic Jul–Aug 2023–2025, 10m GSD. Available via: `https://ftp.natt.is/public/marco/download/sentinel-2-mosaic/`
- **Cultivated land:** Þjóðskrá Íslands, *Ræktað land 2018–2020*. Available via WFS: `http://gis.fasteignaskra.is/geoserver/public/ows`

---

## Citation

If using this pipeline or its outputs, please cite:

> Náttúrufræðistofnun Íslands, 2026. *Automated lupine distribution mapping pipeline*. GitHub: https://github.com/natturufraedistofnun/automated-lupina-mapping

---

## Contact

Náttúrufræðistofnun Íslands (Natural Science Institute of Iceland) — [natt.is](https://natt.is) — natt@natt.is