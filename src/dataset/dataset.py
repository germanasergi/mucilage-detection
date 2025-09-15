import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataset.patches import *

class Sentinel2PatchDataset(Dataset):
    """
    PyTorch Dataset to extract 256x256 patches from Sentinel-2 Zarr files
    based on coordinates provided in a CSV.
    """

    def __init__(self, df, bands, patch_size=256, transform=None):
        """
        Args:
            df (pd.DataFrame): CSV containing columns ["zarr_path", "x", "y", "label"].
            bands (list): List of bands to extract.
            patch_size (int): Size of the patch to extract (patch_size x patch_size).
            transform (callable, optional): Optional transform applied to the patch.
        """
        self.df = df.reset_index(drop=True)
        self.bands = bands
        self.patch_size = patch_size
        self.transform = transform

        # Build an index mapping zarr_path -> list of row indices for faster access
        self.path_groups = {
            z: group.index.tolist() for z, group in self.df.groupby("zarr_path")
        }

        # Flattened list of (zarr_path, row_index) to index dataset
        self.indices = [
            (z, idx) for z, group in self.path_groups.items() for idx in group
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        zarr_path, row_idx = self.indices[idx]
        print(f"Loading patch from {zarr_path}, row {row_idx}")
        row = self.df.loc[row_idx]

        # Open Zarr (datatree) once per patch access
        ds = xr.open_datatree(zarr_path, engine="zarr", mask_and_scale=False, chunks={})
        stack = build_stack_10m(ds, self.bands)  # (H, W, C)

        x, y, label = row["x"], row["y"], row["label"]

        patch = stack.isel(
            y=slice(y, y + self.patch_size),
            x=slice(x, x + self.patch_size)
        ).to_numpy().astype(np.float32)

        ds.close()

        # Skip invalid patches
        if np.isnan(patch).any() or np.isinf(patch).any():
            raise ValueError(f"Invalid patch at {zarr_path}, row {row_idx}")

        # Convert to torch tensor [C, H, W]
        patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.transform:
            patch_tensor = self.transform(patch_tensor)

        return patch_tensor, label_tensor


class Sentinel2NumpyDataset(Dataset):
    def __init__(self, df, bands, patch_size=256, transform=None):
        """
        Load all patches from .zarr files into memory (NumPy) once,
        then serve them as a PyTorch Dataset.

        Args:
            csv_path (str): CSV containing columns [zarr_path, x, y, label].
            bands (list): List of band names to extract.
            patch_size (int): Patch size to extract.
            transform (callable): Optional transform on the patch.
        """
        self.df = df.reset_index(drop=True)
        self.bands = bands
        self.patch_size = patch_size
        self.transform = transform

        self.X, self.y = self._build_numpy_dataset()

    def _build_numpy_dataset(self):
        all_patches = []
        all_labels = []

        # group by zarr_path to avoid reopening each patch
        for zarr_path, group in tqdm(self.df.groupby("zarr_path"), desc="Building NumPy dataset"):
            ds = xr.open_datatree(zarr_path, engine="zarr", mask_and_scale=False, chunks={})
            stack = build_stack_10m(ds, self.bands)

            for _, row in group.iterrows():
                x, y, label = row["x"], row["y"], row["label"]
                patch = stack.isel(
                    y=slice(y, y + self.patch_size),
                    x=slice(x, x + self.patch_size)
                ).to_numpy().astype(np.float32)

                if np.isnan(patch).any() or np.isinf(patch).any():
                    continue 

                all_patches.append(patch)
                all_labels.append(label)

            ds.close()

        X = np.stack(all_patches, axis=0)   # (N, H, W, C)
        y = np.array(all_labels)

        return X, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        patch = self.X[idx]  # (H, W, C)
        label = self.y[idx]

        # Convert to torch tensor [C, H, W]
        patch = torch.from_numpy(patch).permute(2, 0, 1).float()
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            patch = self.transform(patch)

        return patch, label