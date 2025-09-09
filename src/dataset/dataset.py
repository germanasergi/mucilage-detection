import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataset.patches import *

class Sentinel2PatchDataset(Dataset):
    def __init__(self, df, bands, patch_size=256, transform=None):
        """
        Args:
            df (pd.DataFrame): must contain:
                - 'zarr_path' : path to the .zarr file
                - 'x' : top-left col
                - 'y' : top-left row
                - 'label' (optional): class label
            bands (list): list of band names
            patch_size (int): size of patches to extract
            transform (callable): optional transform (augmentations)
        """
        self.df = df.reset_index(drop=True)
        self.bands = bands
        self.patch_size = patch_size
        self.transform = transform
        self.has_labels = "label" in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        print(f"Fetching patch {idx}")
        row = self.df.iloc[idx]
        zarr_path, x, y = row["zarr_path"], row["x"], row["y"]

        # open zarr, build stack
        with xr.open_datatree(zarr_path, engine="zarr", mask_and_scale=False, chunks={}) as ds:
            stack = build_stack_10m(ds, self.bands)
            patch = stack.isel(
                y=slice(y, y + self.patch_size),
                x=slice(x, x + self.patch_size)
            ).to_numpy()

        patch = stack.isel(
            y=slice(y, y + self.patch_size),
            x=slice(x, x + self.patch_size)
        ).to_numpy()
        ds.close()

        # convert to torch tensor [C, H, W]
        patch = torch.from_numpy(patch).permute(2, 0, 1).float()

        if self.transform:
            patch = self.transform(patch)

        if self.has_labels:
            label = torch.tensor(row["label"], dtype=torch.long)
            return patch, label
        else:
            return patch


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
                ).to_numpy().astype(np.float32)  # (H, W, C)

                if np.isnan(patch).any() or np.isinf(patch).any():
                    continue  # skip invalid patches

                all_patches.append(patch)
                all_labels.append(label)

            ds.close()
            print(f"Loaded {len(all_patches)} patches so far...")

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