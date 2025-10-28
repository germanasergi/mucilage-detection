import os
import gc
import random
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from datetime import datetime
import glob
from rasterio.transform import from_origin
from scipy.ndimage import generate_binary_structure, label, binary_fill_holes, binary_erosion

def compute_amei(ds, eps=1e-6):
    red   = ds["measurements/reflectance/r10m/b04"] / 10000.0
    green = ds["measurements/reflectance/r10m/b03"] / 10000.0
    nir   = resample_to_10m(ds, 'b8a', 'b04', folder='measurements') / 10000.0
    swir  = resample_to_10m(ds, 'b11', 'b04', folder='measurements') / 10000.0

    # AMEI = (2*red + nir - 2*swir) / (green + 0.25*swir)
    denom = green + 0.25 * swir
    amei  = (2*red + nir - 2*swir) / (denom + eps)

    return amei.rename("amei")  # keep DataArray with name

def load_era5_sst(ds, bbox, target_res, ref_band, date, pat):
    """
    Load full ERA5 SST, reproject it to Sentinel-2 grid (EPSG:32633),
    and match the full 10980×10980 tile extent.
    """
    era5 = xr.open_dataset(
        f"https://edh:{pat}@data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr",
        chunks={},
        engine="zarr",
    )

    if "sst" not in era5:
        raise ValueError("ERA5 SST variable not found in dataset.")

    sst = era5["sst"]

    # --- Select closest timestamp ---
    if date is not None:
        t = np.datetime64(datetime.fromisoformat(str(date)).replace(tzinfo=None))
        sst = sst.sel(valid_time=t, method="nearest")

    # --- Rename and orient ---
    sst = sst.rename({"longitude": "x", "latitude": "y"}).transpose("y", "x")
    sst = sst.assign_coords(
        x=((sst.x + 180) % 360) - 180
        ).sortby("x")

    # --- Assign CRS and geotransform ---
    sst = sst.rio.write_crs("EPSG:4326")
    res_x = float(sst.x[1] - sst.x[0])
    res_y = float(sst.y[1] - sst.y[0])
    transform = from_origin(
        west=float(sst.x.min()),
        north=float(sst.y.max()),
        xsize=res_x,
        ysize=abs(res_y)
    )
    sst = sst.rio.write_transform(transform)

    # --- Get reference Sentinel-2 band for target grid ---
    ref = ds[f"measurements/reflectance/{target_res}/{ref_band}"].rio.write_crs("EPSG:32633")

    print(f"[ERA5 SST] Original grid: {sst.shape}, CRS=EPSG:4326 → Reprojecting to match Sentinel-2 tile...")

    # --- Try reprojection ---
    try:
        sst_matched = sst.rio.reproject_match(ref)
    except Exception as e:
        print(f"[ERA5 SST] Reprojection failed ({e}) → Using fallback interpolation.")
        sst_matched = sst.interp_like(ref, method="linear")

    # --- Convert from Kelvin to Celsius and fill NaNs ---
    sst_matched = sst_matched - 273.15
    sst_matched = sst_matched.fillna(sst_matched.mean())

    # --- Ensure exact shape match ---
    sst_matched = sst_matched.interp_like(ref, method="nearest")

    print(f"[ERA5 SST] Final shape: {sst_matched.shape}")
    print(f"[ERA5 SST] Value range (°C): {float(sst_matched.min()):.2f} – {float(sst_matched.max()):.2f}")
    print(f"[ERA5 SST] NaN ratio: {float(np.isnan(sst_matched).mean()):.4f}")

    return sst_matched


def clean_water_mask(ds, target_res="r10m"):
    """
    Fix water mask by:
    1. Keeping only the sea (remove lakes/rivers).
    2. Filling cloud holes in the sea.
    """
    scl = resample_band(ds, 'scl', target_res=target_res, ref='b04')
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

def resample_band(ds, band, target_res="r10m", ref="b04", crs="EPSG:32632"):
    """
    Resample any band (reflectance or classification) to target resolution.
    """
    ref_band = ds[f"measurements/reflectance/{target_res}/{ref}"].rio.write_crs(crs) # Reference band at target resolution

    if band == "scl":
        band_da = ds[f"conditions/mask/l2a_classification/r20m/{band}"].rio.write_crs(crs)
        source_res = "r20m"
    else:
        # Detect which reflectance resolution contains the band
        source_res = next(
        (r for r in ["r10m", "r20m", "r60m"] if band in ds[f"measurements/reflectance/{r}"]),
        None
        )
        if source_res is None:
            raise ValueError(f"Band {band} not found in reflectance or scl folder")
        band_da = ds[f"measurements/reflectance/{source_res}/{band}"].rio.write_crs(crs)
    # If source == target, no resampling needed
    if source_res == target_res:
        return band_da

    return band_da.rio.reproject_match(ref_band)


def build_stack(ds, bands, target_res="r10m", ref_band="b04", crs="EPSG:32632", bbox=None, date=None, pat=None):
    """
    Build a lazy dask-backed (H, W, C) stack from bands, resampling as needed.

    Args:
        ds: xarray Dataset or DataTree
        bands: list of band names to include
        target_res: desired output resolution for all bands
        ref_band: reference band for resampling (default: 'b04' red)
        crs: CRS to assign if missing

    Returns:
        xarray.DataArray with dimensions (y, x, band)
    """
    stack = []

    for b in bands:
        if b in ds['measurements/reflectance/r10m'] or \
           b in ds['measurements/reflectance/r20m'] or \
           b in ds['measurements/reflectance/r60m']:
            arr = resample_band(ds, b, target_res=target_res, ref=ref_band, crs=crs) / 10000.0
        # elif b == "amei":
        #     arr = compute_amei(ds)
        elif b == "sst":
            arr = load_era5_sst(ds, bbox=bbox, target_res=target_res, ref_band=ref_band, date=date, pat=pat)
            print(f"SST min/max loaded: {float(arr.min().values)}, {float(arr.max().values)}")
        else:
            raise ValueError(f"Band {b} not found or not supported.")

        # Expand dims for stacking
        arr = arr.expand_dims(band=[b])
        stack.append(arr)

    # Concatenate all bands along 'band' dimension
    stacked = xr.concat(stack, dim="band").transpose("y", "x", "band")
    return stacked


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
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, "data/adr_test/target")
    zarr_files = glob.glob(os.path.join(DATA_DIR, "*.zarr"))

    # Fix seed
    np.random.seed(42)

    # Create patches DataFrame
    df_patches = create_patches_dataframe(zarr_files, n_patches_per_file=100, patch_size=256)
    df_patches.to_csv(os.path.join(BASE_DIR, "csv/patches.csv"), index=False)

if __name__ == "__main__":
    main()