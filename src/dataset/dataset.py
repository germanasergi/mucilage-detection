import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import pandas as pd
from patches import *

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
        row = self.df.iloc[idx]
        zarr_path, x, y = row["zarr_path"], row["x"], row["y"]

        # open zarr, build stack
        ds = xr.open_datatree(zarr_path, engine="zarr", mask_and_scale=False, chunks={})
        stack = build_stack_10m(ds, self.bands)
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