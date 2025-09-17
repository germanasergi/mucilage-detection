import os
import s3fs
import zarr
import xarray as xr
from icechunk.repository import Repository
from tqdm import tqdm

def safe_to_icechunk(s3_url, repo_path, bands=None):
    """
    Convert a SAFE on S3 to Zarr and store in Icechunk repo.
    
    s3_url: S3 URL to SAFE, e.g., "s3://bucket/path/to/S2A_MSIL2A_...SAFE"
    repo_path: Icechunk repo path
    bands: list of bands to include
    """
    # Open SAFE remotely
    fs = s3fs.S3FileSystem(anon=False)  # needs credentials from environment
    store = s3fs.S3Map(root=s3_url, s3=fs, check=False)  # gives zarr-like interface
    
    # Open with xarray if SAFE is cloud-optimized
    ds = xr.open_dataset(store, engine="zarr", mask_and_scale=False, chunks={})
    
    # Select bands if needed
    if bands is not None:
        ds = ds[bands]

    # Prepare Zarr store path inside Icechunk repo
    os.makedirs(repo_path, exist_ok=True)
    zarr_name = os.path.basename(s3_url).replace(".SAFE", ".zarr")
    zarr_path = os.path.join(repo_path, zarr_name)

    # Write to Zarr (memory or disk-backed)
    ds.to_zarr(zarr_path, mode="w")
    ds.close()

    # Add to Icechunk repo
    if not os.path.exists(os.path.join(repo_path, ".icechunk")):
        repo = Repository.init(repo_path)
    else:
        repo = Repository(repo_path)
    repo.add_dataset(zarr_path)
    repo.commit(f"Added dataset from {s3_url}")

