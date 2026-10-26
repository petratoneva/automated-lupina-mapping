# ── helpers_s2.py ─────────────────────────────────────────────────────────────
# Sentinel-2 specific helpers and feature engineering.
# Reuses shared helpers from helpers.py where possible.
# Import in all S2 notebooks: from helpers_s2 import *

# ── Shared imports from aerial helpers ────────────────────────────────────────
from helpers import (
    safe_download,
    rasterize_vector,
    apply_chunked,
    merge_batch,
)

import gc
import os
import numpy as np
import rasterio
from rasterio.features import rasterize, shapes
import geopandas as gpd
from shapely.geometry import box, shape
from scipy.ndimage import median_filter, binary_opening, binary_closing


# ── S2 feature engineering ────────────────────────────────────────────────────
# 15 spectral features — no texture (NIR_contrast not meaningful at 10m)
# Bands order: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12

FEATURE_NAMES_S2 = [
    "B2", "B3", "B4", "B8",         # raw 10m bands
    "B5", "B8A", "B11",             # raw 20m bands (resampled to 10m)
    "NDVI", "NDRE", "CIre",         # vegetation indices
    "EVI", "NDWI", "NDMI",          # vegetation + moisture
    "SAVI", "SWIR_ratio",           # soil + SWIR
]

def engineer_features_s2(img, nodata_val=32767, scale=10000.0):
    """
    Input:  img (10, H, W) int16 raw DN, bands [B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12]
    Output: (H*W, 15) float32
    Nodata pixels set to 0.
    """
    img_f = img.astype(np.float32)
    img_f[img_f >= nodata_val] = np.nan
    img_f = img_f / scale

    B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 = [img_f[i] for i in range(10)]
    eps = 1e-6

    # Vegetation indices
    NDVI       = (B8  - B4)  / (B8  + B4  + eps)
    NDRE       = (B8A - B5)  / (B8A + B5  + eps)   # red-edge NDVI
    CIre       = (B7  / (B5  + eps)) - 1             # chlorophyll red-edge
    EVI        = 2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1 + eps)

    # Water / moisture
    NDWI       = (B3  - B8)  / (B3  + B8  + eps)
    NDMI       = (B8A - B11) / (B8A + B11 + eps)

    # Soil adjusted
    SAVI       = 1.5 * (B8 - B4) / (B8 + B4 + 0.5 + eps)

    # SWIR ratio
    SWIR_ratio = B11 / (B12 + eps)

    features = np.stack([
        B2, B3, B4, B8,
        B5, B8A, B11,
        NDVI, NDRE, CIre,
        EVI, NDWI, NDMI,
        SAVI, SWIR_ratio,
    ], axis=0)

    features = np.nan_to_num(features, nan=0.0)
    C, H, W  = features.shape
    return features.reshape(C, -1).T


# ── S2 nodata handling ────────────────────────────────────────────────────────
def mask_nodata_s2(img, nodata_val=32767):
    """Returns boolean mask — True where pixel is valid (not nodata)."""
    return (img[0] < nodata_val) & (img[0] > 0)


# ── S2 postprocessing — cultivated suppression only ───────────────────────────
def postprocess_tile_s2(binary, profile, cultivated_gdf, cult_buffer_m=20):
    """
    Apply cultivated land suppression to a single S2 binary tile.
    cult_buffer_m: buffer around cultivated polygons in metres (captures edges)
    """
    tile_crs    = profile["crs"]
    tile_bounds = rasterio.transform.array_bounds(
        profile["height"], profile["width"], profile["transform"]
    )
    tile_box  = box(*tile_bounds)
    cult_clip = cultivated_gdf[cultivated_gdf.to_crs(tile_crs).intersects(tile_box)].copy()

    if len(cult_clip) > 0:
        cult_clip = cult_clip.to_crs(tile_crs)
        # Buffer by 2 pixels (20m at 10m resolution) to capture field edges
        cult_clip["geometry"] = cult_clip.geometry.buffer(cult_buffer_m)
        cult_mask = rasterize_vector(cult_clip, profile)
        binary[cult_mask] = 0
        del cult_mask

    del cult_clip; gc.collect()
    return binary


# ── S2 vectorization ──────────────────────────────────────────────────────────
def vectorize_tile_s2(binary, transform, crs, pixel_area_m2, min_m2=1):
    """
    Vectorize a binary S2 tile.
    Returns GeoDataFrame with polygons, or None if no patches.
    """
    polygons = [
        {"geometry": shape(geom), "value": int(val)}
        for geom, val in shapes(binary.astype(np.uint8), transform=transform)
        if int(val) == 1
    ]
    if not polygons:
        return None

    gdf = gpd.GeoDataFrame(polygons, crs=crs)
    gdf["area_m2"] = gdf.geometry.area.round(2)
    gdf["area_ha"] = (gdf["area_m2"] / 10000).round(4)
    gdf = gdf[gdf["area_m2"] >= min_m2].reset_index(drop=True)

    if len(gdf) == 0:
        return None

    gdf["patch_id"] = gdf.index + 1
    return gdf
