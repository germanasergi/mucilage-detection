import os
import gc
import random
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import glob
from scipy.ndimage import generate_binary_structure, label, binary_fill_holes, binary_erosion

def resample_to_10m(ds, band, ref, folder):
    """
    Resample band to match the resolution & grid of reference band.
    ds: opened .zarr datatree
    band: name of band to resample (string)
    ref: name of reference band (string)
    """
    crs_code = "EPSG:32632"

    # Define reference band
    ref_band = ds[f"measurements/reflectance/r10m/{ref}"]  # reference (10m red)
    ref_band = ref_band.rio.write_crs(crs_code, inplace=True)

    # Band to convert
    if folder == 'measurements':
        band_20m = ds[f"measurements/reflectance/r20m/{band}"]
    else:
        band_20m = ds[f"conditions/mask/l2a_classification/r20m/{band}"] # for classification band
    band_10m = band_20m.rio.write_crs(crs_code, inplace=True)  # ensure CRS

    return band_10m.rio.reproject_match(ref_band)

def compute_amei(ds, eps=1e-6):
    red   = ds["measurements/reflectance/r10m/b04"] / 10000.0
    green = ds["measurements/reflectance/r10m/b03"] / 10000.0
    nir   = resample_to_10m(ds, 'b8a', 'b04', folder='measurements') / 10000.0
    swir  = resample_to_10m(ds, 'b11', 'b04', folder='measurements') / 10000.0

    # AMEI = (2*red + nir - 2*swir) / (green + 0.25*swir)
    denom = green + 0.25 * swir
    amei  = (2*red + nir - 2*swir) / (denom + eps)

    return amei.rename("amei")  # keep DataArray with name



def clean_water_mask(ds):
    """
    Fix water mask by:
    1. Keeping only the sea (remove lakes/rivers).
    2. Filling cloud holes in the sea.
    """

    scl = resample_to_10m(ds, 'scl', 'b04', folder='conditions')
    scl = scl.squeeze().values
    raw_water_mask = (scl == 6)
    H, W = raw_water_mask.shape

    # Keep only the largest connected component that touches border (most prob the sea)
    st = generate_binary_structure(2, 2)   # 8-connectivity
    lab, nlab = label(raw_water_mask, structure=st) # different label for each connected water body
    
    if nlab == 0:
        return np.zeros_like(raw_water_mask, dtype=bool)

    # Find component sizes
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0  # background, we don't consider it

    # Keep the largest component
    largest_label = sizes.argmax()
    sea_only = (lab == largest_label)

    # Fill holes inside sea (caused by clouds)
    sea_filled = binary_fill_holes(sea_only)

    return sea_filled.astype(bool)


def build_stack_10m(ds, bands):
    """
    Return a lazy dask-backed stack (H, W, C) instead of full NumPy.
    """
    stack = []

    for b in bands:
        if b in ds['measurements/reflectance/r10m']:
            arr = ds['measurements/reflectance/r10m'][b] / 10000.0
        elif b in ds['measurements/reflectance/r20m']:
            arr = resample_to_10m(ds, b, 'b04', folder='measurements') / 10000.0
        elif b == "amei":
            arr = compute_amei(ds)
        else:
            raise ValueError(f"Band {b} not found or not supported.")

        # Expand dims and assign band coordinate for all arrays
        arr = arr.expand_dims(band=[b])
        stack.append(arr)

    # Concatenate along band dimension lazily
    stack = xr.concat(stack, dim="band").transpose("y", "x", "band")
    return stack


def sample_patch_corners(water_mask, n_patches, patch_size=256, border_weight=0.6):
    """Sample patch *corners* (top-left) biased toward shoreline."""
    H, W = water_mask.shape
    border_mask = water_mask & ~binary_erosion(water_mask, iterations=300)

    probs = np.zeros_like(water_mask, dtype=np.float32)
    probs[border_mask] = border_weight
    probs[water_mask]  = 1.0 - border_weight
    probs = probs / probs.sum()

    flat_idx = np.random.choice(H*W, size=n_patches, replace=False, p=probs.ravel())
    centers = np.column_stack(np.unravel_index(flat_idx, (H, W)))

    # convert centers → corners (top-left coordinates)
    half = patch_size // 2
    corners = [(max(0, r-half), max(0, c-half)) for r, c in centers]
    return corners


def create_patches_dataframe(zarr_files, n_patches_per_file=150, patch_size=256):
    """
    For each zarr file, extract top-left coordinates of sampled patches,
    and store them along with zarr path in a DataFrame.
    
    Returns:
        df_patches: DataFrame with columns ['zarr_path', 'x', 'y']
    """
    records = []

    for zf in zarr_files:
        print(f"Processing {zf} for patch coordinates...")
        ds = xr.open_datatree(zf, engine="zarr", mask_and_scale=False, chunks={})

        # Compute water mask
        water_mask = clean_water_mask(ds)
        
        # Retrieve shape bands
        band = ds['measurements/reflectance/r10m/b04']
        H, W = band.shape

        corners = sample_patch_corners(water_mask, n_patches=n_patches_per_file, patch_size=patch_size)
        
        for y, x in corners:
            # Ensure the patch fits within the image
            if y + patch_size <= H and x + patch_size <= W:
                records.append({'zarr_path': zf, 'x': x, 'y': y})
        
        # Free memory
        del ds
        gc.collect()
    
    df_patches = pd.DataFrame(records)
    print(f"Total patches collected: {len(df_patches)}")

    return df_patches

def main():
    # Directory containing .zarr files
    BASE_DIR = "/home/ubuntu/mucilage_pipeline/mucilage-detection"
    DATA_DIR = os.path.join(BASE_DIR, "data/adr_test/target")
    zarr_files = glob.glob(os.path.join(DATA_DIR, "*.zarr"))

    # Fix seed
    np.random.seed(42)

    # Create patches DataFrame
    df_patches = create_patches_dataframe(zarr_files, n_patches_per_file=100, patch_size=256)
    df_patches.to_csv(os.path.join(BASE_DIR, "csv/patches.csv"), index=False)

if __name__ == "__main__":
    main()