import os
import pandas as pd
import numpy as np
from requests import patch
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F


def plot_metrics(
    df,
    bands=['B02', 'B03', 'B04'],
    log_scale=False,
    title="SSIM Metrics Over Training Epochs",
    y_label="SSIM",
    verbose=False,
    save=False,
    save_path="./",
    color_palette="plasma",
):
    """
    Plot training and validation metrics for multiple spectral bands.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing metrics data with 'epoch' column and metrics columns
        formatted as 'train_{band}' and 'val_{band}'.
    bands : list, optional
        List of band names to plot, default is ['B02', 'B03', 'B04'].
    log_scale : bool, optional
        Whether to use logarithmic scale for y-axis, default is False.
    title : str, optional
        Plot title, default is "SSIM Metrics Over Training Epochs".
    y_label : str, optional
        Y-axis label, default is "SSIM".
    verbose : bool, optional
        Whether to display the plot, default is False.
    save : bool, optional
        Whether to save the plot, default is False.
    save_path : str, optional
        Directory path to save the figure, default is "./".
    color_palette : str, optional
        Name of seaborn color palette to use, default is "plasma".

    Returns
    -------
    None
        Function creates and optionally saves/displays a plot.
    """
    # Set up color palette
    colors = sns.color_palette(color_palette, len(bands))

    # Create figure
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Loop through bands to plot
    for i, band in enumerate(bands):
        # Plot training curves (dashed)
        ax.plot(
            df['epoch'],
            df[f'train_{band}'],
            '--',
            label=f'Train {band}',
            color=colors[i],
        )

        # Plot validation curves (solid)
        ax.plot(
            df['epoch'],
            df[f'val_{band}'],
            label=f'Val {band}',
            color=colors[i],
        )

    # Apply log scale if requested
    if log_scale:
        ax.set_yscale('log')

    # Add labels and title
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Set tight layout for better spacing
    plt.tight_layout()

    # Save plot if requested
    if save:
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Construct filename based on metric type
        filename = f"{y_label.lower().replace(' ', '_')}_metric_training.svg"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')

        if verbose:
            print(f"Plot saved to: {full_path}")

    # Display plot if requested
    if verbose:
        plt.show()

    plt.close()


def plot_training_loss(
    df,
    title="Training and Validation Loss",
    y_label="Loss",
    log_scale=False,
    verbose=False,
    save=False,
    save_path="./",
    color_palette="RdBu_r",
):
    """
    Plot training and validation loss over epochs.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'epoch', 'train_loss', and 'val_loss' columns.
    title : str, optional
        Plot title, default is "Training and Validation Loss".
    y_label : str, optional
        Y-axis label, default is "Loss".
    log_scale : bool, optional
        Whether to use logarithmic scale for y-axis, default is False.
    verbose : bool, optional
        Whether to display the plot, default is False.
    save : bool, optional
        Whether to save the plot, default is False.
    save_path : str, optional
        Directory path to save the figure, default is "./".
    color_palette : str, optional
        Name of seaborn color palette to use, default is "plasma".

    Returns
    -------
    None
        Function creates and optionally saves/displays a plot.
    """
    # Set up color palette
    colors = sns.color_palette(color_palette, 2)

    # Create figure
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    ax.plot(df["epoch"], df["train_loss"], label='Training Loss', color=colors[0])
    ax.plot(df["epoch"], df["val_loss"], label='Validation Loss', color=colors[1])

    # Apply log scale if requested
    if log_scale:
        ax.set_yscale('log')

    # Add labels and title
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Set tight layout for better spacing
    plt.tight_layout()

    # Save plot if requested
    if save:
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Construct filename
        filename = "loss_plot.svg"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')

        if verbose:
            print(f"Plot saved to: {full_path}")

    # Display plot if requested
    if verbose:
        plt.show()

    plt.close()


def plot_all_chunks(data_tree, band, res, chunk_size_y, chunk_size_x, nb_chunks_y, nb_chunks_x, cmap="viridis", verbose= True, figsize_scale=3):
    """
    Plot all chunks of a given band and resolution.

    Parameters:
    - data_tree: xarray Dataset (e.g. dt.measurements.reflectance.r20m)
    - band: str, band name (e.g. "b05")
    - cmap: str, matplotlib colormap
    - figsize_scale: int, scales the figure size (default is 3)
    """
    res_key = f"r{res}"
    y_res = 'y' #MODIFIED
    x_res = 'x'
    data_tree = data_tree.measurements.reflectance[res_key]
    # Set up plot grid
    fig, axes = plt.subplots(
        nb_chunks_y, nb_chunks_x,
        figsize=(figsize_scale * nb_chunks_x, figsize_scale * nb_chunks_y)
    )

    # Plot each chunk
    for i in range(nb_chunks_y):
        for j in range(nb_chunks_x):
            ax = axes[i, j] if nb_chunks_y > 1 else axes[j]
            y_start = i * chunk_size_y
            x_start = j * chunk_size_x
            chunk = data_tree[band].isel(
                {y_res: slice(y_start, y_start + chunk_size_y),
                 x_res: slice(x_start, x_start + chunk_size_x)}
            ).load()
            ax.imshow(chunk, cmap=cmap, vmin=float(chunk.min()), vmax=float(chunk.max()))
            ax.set_title(f"Chunk ({i},{j})")
            ax.axis("off")
    if verbose:
        plt.tight_layout()
        plt.savefig("chunks_plot.png", dpi=300)

def get_rgb(patch):
    """
    Convert multi-band patch to RGB for visualization.
    bands: tuple with indices for (R, G, B).
    """
    rgb = patch[:, :, [3, 2, 1]] #now there is b1 as well
    p2, p98 = np.nanpercentile(rgb, (2, 98))
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
    
    return rgb


def save_attention(attn_map, patch_img, upsample_size=256, save_path="attention.png"):
    """
    Show RGB patch and its attention map side by side.

    attn_map: [H, W] attention map (torch.Tensor or np.ndarray)
    patch_img: [H0, W0, C] original patch (numpy or tensor)
    """
    # --- Normalize attention ---
    attn_map = attn_map.detach().cpu().numpy() if torch.is_tensor(attn_map) else attn_map
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # --- Upsample to patch size ---
    attn_tensor = torch.tensor(attn_map).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    attn_up = F.interpolate(attn_tensor, size=(upsample_size, upsample_size),
                            mode="bilinear", align_corners=False)
    attn_up = attn_up.squeeze().numpy()

    # --- Get RGB composite ---
    rgb = get_rgb(patch_img)  # your helper function (e.g. bands [b04,b03,b02])

    # --- Plot side by side ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(rgb)
    axes[0].set_title("RGB Patch")
    axes[0].axis("off")

    im = axes[1].imshow(attn_up, cmap="jet")
    axes[1].set_title("Attention Map")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="Attention weight")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_multi_attention(attn_maps, patch_img, upsample_size=256, save_path="multi_attention.png", threshold=0.5):
    rgb = get_rgb(patch_img)
    num_heads = len(attn_maps)

    # Create figure: RGB + each head + aggregated mask
    fig, axes = plt.subplots(1, num_heads + 2, figsize=(4 * (num_heads + 2), 4))

    # --- RGB image ---
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")

    # --- Per-head attention maps ---
    attn_list = []
    for i, attn in enumerate(attn_maps):
        attn = attn.detach().cpu().numpy()
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        attn_tensor = torch.tensor(attn).unsqueeze(0).unsqueeze(0)
        attn_up = F.interpolate(attn_tensor, size=(upsample_size, upsample_size),
                                mode="bilinear", align_corners=False)
        attn_up = attn_up.squeeze().numpy()
        attn_list.append(attn_up)

        axes[i + 1].imshow(attn_up, cmap="jet")
        axes[i + 1].set_title(f"Head {i+1}")
        axes[i + 1].axis("off")

    # --- Aggregated binary mask (mean across heads) ---
    agg_map = np.max(np.stack(attn_list, axis=0), axis=0)
    binary_mask = (agg_map >= threshold).astype(np.uint8)

    axes[num_heads + 1].imshow(binary_mask, cmap="gray")
    axes[num_heads + 1].set_title("Binary mask")
    axes[num_heads + 1].axis("off")

    plt.savefig(save_path, dpi=150)
    plt.close()