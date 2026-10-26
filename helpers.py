# ── helpers.py ────────────────────────────────────────────────────────────────
# Shared imports, helper functions and feature engineering.
# Import in all scripts: from helpers import *

import gc
import os
import time
import requests
import multiprocessing
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask as rio_mask
from rasterio.features import rasterize, shapes
from rasterio.transform import from_origin
from scipy.ndimage import uniform_filter, median_filter, binary_opening, binary_closing
import geopandas as gpd
import joblib
from shapely.geometry import box, shape


# ── Download ──────────────────────────────────────────────────────────────────
def safe_download(url, out_path, retries=3):
    """Download a file with retries. Returns True on success."""
    for attempt in range(retries):
        try:
            r = requests.get(url, stream=True, timeout=60)
            if r.status_code == 200:
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        f.write(chunk)
                return True
            else:
                print(f"  [!] HTTP {r.status_code}: {os.path.basename(out_path)}")
        except Exception as e:
            if attempt < retries - 1:
                print(f"  [!] Retry {attempt+1}/{retries}: {e} — waiting …")
                time.sleep(5)
            else:
                print(f"  [!] Download failed after {retries} attempts: {e}")
    return False


# ── Rasterization ─────────────────────────────────────────────────────────────
def rasterize_vector(gdf, ref_profile):
    """Rasterize a GeoDataFrame to match a reference raster grid. Returns bool array."""
    if len(gdf) == 0:
        return np.zeros((ref_profile["height"], ref_profile["width"]), dtype=bool)
    return rasterize(
        [(geom, 1) for geom in gdf.geometry if geom is not None],
        out_shape=(ref_profile["height"], ref_profile["width"]),
        transform=ref_profile["transform"],
        fill=0, dtype=np.uint8, all_touched=False,
    ).astype(bool)


# ── Morphological cleanup ─────────────────────────────────────────────────────
def apply_chunked(binary, func, kernel_size, chunk=3000):
    """Apply morphological function in horizontal chunks to save RAM."""
    if kernel_size == 0:
        return binary.copy()
    H, W    = binary.shape
    result  = np.zeros((H, W), dtype=np.uint8)
    struct  = np.ones((kernel_size, kernel_size))
    overlap = kernel_size
    for row in range(0, H, chunk):
        row_start = max(0, row - overlap)
        row_end   = min(H, row + chunk + overlap)
        strip     = func(binary[row_start:row_end], structure=struct) \
                    if func != median_filter \
                    else median_filter(binary[row_start:row_end], size=kernel_size)
        out_start = row - row_start
        out_end   = out_start + min(chunk, H - row)
        result[row:row + min(chunk, H - row)] = strip[out_start:out_end]
    return result


# ── Per-tile postprocessing ───────────────────────────────────────────────────
def postprocess_tile(binary, profile, cultivated_gdf, shadow_path=None,
                     median_size=3, opening_size=4, closing_size=5):
    """
    Apply cultivated suppression, shadow mask and morphological cleanup to a tile.
    Returns cleaned binary array (H, W) uint8.
    """
    tile_crs    = profile["crs"]
    tile_bounds = rasterio.transform.array_bounds(
        profile["height"], profile["width"], profile["transform"]
    )
    tile_box = box(*tile_bounds)

    # [1] Cultivated land suppression
    cult_clip = cultivated_gdf[cultivated_gdf.to_crs(tile_crs).intersects(tile_box)].copy()
    if len(cult_clip) > 0:
        cult_clip = cult_clip.to_crs(tile_crs)
        cult_mask = rasterize_vector(cult_clip, profile)
        binary[cult_mask] = 0
        del cult_mask
    del cult_clip; gc.collect()

    # [2] Shadow mask
    if shadow_path and os.path.exists(shadow_path):
        with rasterio.open(shadow_path) as src:
            shadow = src.read(1).astype(bool)
        binary[shadow] = 0
        del shadow; gc.collect()

    # [3] Morphological cleanup
    if median_size > 0:
        binary = median_filter(binary, size=median_size)
    if opening_size > 0:
        binary = binary_opening(binary,
            structure=np.ones((opening_size, opening_size))).astype(np.uint8)
    if closing_size > 0:
        binary = binary_closing(binary,
            structure=np.ones((closing_size, closing_size))).astype(np.uint8)

    return binary


# ── Snap tile transform to grid ───────────────────────────────────────────────
def snap_transform(bounds, res):
    """Return a transform snapped to exact grid bounds."""
    minx, miny, maxx, maxy = bounds
    return from_origin(minx, maxy, res, res)


# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(img, use_nir_contrast=True):
    R, G, B, NIR = img[0], img[1], img[2], img[3]
    eps = 1e-6
    R_n, G_n, B_n, NIR_n = R/255.0, G/255.0, B/255.0, NIR/255.0
    NDVI  = (NIR_n - R_n)   / (NIR_n + R_n   + eps)
    NDWI  = (G_n   - NIR_n) / (G_n   + NIR_n + eps)
    denom   = NIR_n + 6*R_n - 7.5*B_n + 1
    EVI = np.clip(2.5 * (NIR_n - R_n) / np.maximum(np.abs(denom), 0.1), -1, 2)  
    # EVI is clipped to the range [−1, 2] to remove computation artefacts arising from near-zero denominators

    feature_list = [R_n, G_n, B_n, NIR_n, NDVI, NDWI, EVI]

    if use_nir_contrast:
        def local_contrast(band, size=5):
            mean  = uniform_filter(band.astype(np.float64), size=size)
            mean2 = uniform_filter((band.astype(np.float64))**2, size=size)
            return np.sqrt(np.maximum(mean2 - mean**2, 0)).astype(np.float32)
        feature_list.append(local_contrast(NIR_n, size=5))
        feature_list.append(local_contrast(NIR_n, size=25))

    features = np.stack(feature_list, axis=0)
    C, H, W  = features.shape
    return features.reshape(C, -1).T


# ── Mosaic helper ─────────────────────────────────────────────────────────────
def merge_batch(tile_paths, out_path, dtype="uint8"):
    """Merge a batch of tiles into one tif."""
    src_files         = [rasterio.open(p) for p in tile_paths]
    mosaic, mosaic_tf = merge(src_files, method="first")
    profile           = src_files[0].profile.copy()
    profile.update(height=mosaic.shape[1], width=mosaic.shape[2],
                   transform=mosaic_tf, count=1, dtype=dtype,
                   compress="lzw", tiled=True, blockxsize=256, blockysize=256)
    for s in src_files: s.close()
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic)
    del mosaic; gc.collect()
    return out_path
