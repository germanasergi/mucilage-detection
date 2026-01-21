import os
import argparse
import torch
import numpy as np
import pyproj
import cv2
import xarray as xr
import matplotlib.pyplot as plt
import torch.nn.functional as F

from model_zoo.models import CNN, MILResNet, MILResNetMultiHead, build_timm_model
from utils.plot import save_attention, save_multi_attention, get_rgb
from dataset.patches import resample_band
from dataset.loader import define_model
import yaml
import timm

def latlon_to_pixel(zarr_file, lat, lon):
    # Find CRS of the dataset
    ds = xr.open_datatree(zarr_file, engine="zarr", mask_and_scale=False)
    crs = ds.attrs.get("other_metadata", {}).get("horizontal_CRS_code")

    # Transformer lat/lon → UTM
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x_utm, y_utm = transformer.transform(lon, lat)

    # Find nearest pixel indices
    band = ds['measurements/reflectance/r10m/b04']
    xc = np.argmin(np.abs(band.x.values - x_utm))
    yc = np.argmin(np.abs(band.y.values - y_utm))

    return yc, xc

def centered_patch_from_zarr(zarr_path, yc, xc, patch_size=256, bands=None):
    ds = xr.open_datatree(zarr_path, engine="zarr", mask_and_scale=False)
    crs = ds.attrs.get("other_metadata", {}).get("horizontal_CRS_code")
    ref = ds['measurements/reflectance/r10m/b04']
    H, W = ref.shape

    half = patch_size // 2
    y0 = max(0, yc - half)
    x0 = max(0, xc - half)
    y1 = min(H, y0 + patch_size)
    x1 = min(W, x0 + patch_size)

    patch_bands = []
    for b in bands:
        if b in ds['measurements/reflectance/r10m'] or \
           b in ds['measurements/reflectance/r20m'] or \
           b in ds['measurements/reflectance/r60m']:
            arr = resample_band(ds, b, target_res="r10m", ref="b04", crs=crs) / 10000.0
            patch_bands.append(arr[y0:y1, x0:x1].values)

    patch = np.stack(patch_bands, axis=-1)

    # pad if patch smaller at the edge
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        pad_y = patch_size - patch.shape[0]
        pad_x = patch_size - patch.shape[1]
        patch = np.pad(patch, ((0,pad_y),(0,pad_x),(0,0)), mode="reflect")

    return patch, (y0, x0)


def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def load_model(config, ckpt, device):
    in_channels = len(config['DATASET']['bands'])
    # base_name = config['MODEL']['model_name'].replace("MIL_", "")
    # model = MILResNetMultiHead(
    #     model_name=base_name,
    #     in_channels=in_channels,
    #     num_classes=config['MODEL'].get('num_classes', 2),
    #     pretrained=False,  # no need for pretrained during inference
    #     num_heads=config['MODEL'].get('num_heads', 4)
    # )

    model = define_model(
        name=config['MODEL']['model_name'],
        encoder_name=config['MODEL']['encoder_name'],
        encoder_weights = config['MODEL']['encoder_weights'],
        in_channel=len(config['DATASET']['bands']),
        out_channels=config['MODEL']['num_classes'],
        activation=config['MODEL']['activation'])
    
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    return model


def preprocess_patch(patch_array, mean, std):
    """
    Normalize patch using training statistics.
    patch_array: np.ndarray [H, W, C]
    """
    print("Patch min/max before norm:", np.nanmin(patch_array), np.nanmax(patch_array))
    return (patch_array - mean) / (std + 1e-8)

def get_rgb(patch):
    """
    Convert multi-band patch to RGB for visualization.
    bands: tuple with indices for (R, G, B).
    """
    rgb = patch[:, :, [3, 2, 1]] #now there is b1 as well
    p2, p98 = np.nanpercentile(rgb, (2, 98))
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
    
    return rgb

def show_amei(patch, eps=1e-6, ax=None):
    green = patch[:, :, 2]
    red = patch[:,:,3]
    nir = patch[:,:,7]
    swir = patch[:,:,10]
    denom = green + 0.25 * swir
    amei  = (2*red + nir - 2*swir) / (denom + eps)

    p2, p98 = np.nanpercentile(amei, (2, 98))
    amei = np.clip((amei - p2) / (p98 - p2), 0, 1)
    cmap = plt.cm.turbo

    if ax is None:
        ax = plt.gca()
    ax.imshow(amei, cmap=cmap)
    ax.axis("off")
    return ax


def run_inference(patch_file, model, device, mean, std, task = "classification", save_dir="inference_outputs"):
    os.makedirs(save_dir, exist_ok=True)

    # Load patch (assume npy [H, W, C])
    # patch = np.load(patch_file)  # e.g. shape [H, W, C]
    # patch = patch['X'][0]
    patch = preprocess_patch(patch_file, mean, std)
    # count nan/inf
    if np.isnan(patch).any() or np.isinf(patch).any():
        # fill with mean
        patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
        print("Patch contains NaN or Inf values. Skipping inference.")
        #return

    # Torch tensor
    inp = torch.tensor(patch, dtype=torch.float).permute(2, 0, 1).unsqueeze(0).to(device)  # [1, C, H, W]

    # Forward pass
    if task == "classification":
        with torch.no_grad():
            logits= model(inp)
            #probs = torch.softmax(logits, dim=1)[0, 1].item() #MIL
            #pred = int(probs >= 0.7)
            probs = torch.softmax(logits, dim=1)[:, 1, :, :] #Segmentation
            pred = (probs > 0.7).float() 

        print(f"Prediction: {pred} | Prob(mucilage)={probs:.3f}")

    elif task == "segmentation":
        with torch.no_grad():
            logits = model(inp)
            if logits.shape[1] == 1:
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=1)[:, 1, :, :].squeeze().cpu().numpy()

            pred = (probs > 0.5).astype(np.uint8)

        print(f"Segmentation done")
        
        # Overlay mask on RGB
        rgb = get_rgb(patch)
        overlay = rgb.copy()
        mask = pred.astype(bool)

        # Define a strong red color overlay
        red_mask = np.zeros_like(rgb)
        red_mask[..., 0] = 1.0  # pure red
        alpha = 0.6
        overlay = np.where(mask[..., None], (1 - alpha) * rgb + alpha * red_mask, rgb)

        fig, axes = plt.subplots(1, 4, figsize=(12, 4))
        axes[0].imshow(rgb)
        axes[0].set_title("Input RGB")
        axes[0].axis("off")

        axes[1].imshow(probs, cmap="viridis")
        axes[1].set_title("Predicted Probability")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay (Red = mucilage)")
        axes[2].axis("off")

        # AMEI visualization
        show_amei(patch, ax=axes[3])
        axes[3].set_title("AMEI")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "segmentation_result_2025.png"), dpi=200)
        plt.show()

    # # Save attention maps
    # if pred == 1:  # only for mucilage
    #     patch_img = patch  # already normalized → convert back to RGB for display
    #     save_path = os.path.join(save_dir, "attention_tuj.png")
    #     per_sample_maps = [head[0] for head in attn_maps]  # extract from batch
    #     save_multi_attention(per_sample_maps, patch_img, save_path=save_path)
    #     print(f"Saved attention maps → {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single patch")
    #parser.add_argument("--patch_file", type=str, required=True, help="Path to input .npy patch file [H,W,C]")
    parser.add_argument("--zarr_file", type=str, default="/home/ubuntu/mucilage_pipeline/mucilage-detection/data/adr_inference/target/S2A_MSIL2A_20250809T100041_N0511_R122_T32TQQ_20250809T122414.zarr", help="Path to input .zarr file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model .pth")
    parser.add_argument("--task", type=str, default="classification", help="Task type: classification or segmentation")
    parser.add_argument("--lat", type=float, default=44.8, help="Latitude of patch center")
    parser.add_argument("--lon", type=float, default=12.6, help="Longitude of patch center")
    args = parser.parse_args()

    # Crop patch
    bands = ['b01','b02', 'b03', 'b04', 'b05', 'b06', 'b07','b08','b8a', 'b11', 'b12']
    yc, xc = latlon_to_pixel(args.zarr_file, lat=args.lat, lon=args.lon)
    patch, (y0, x0) = centered_patch_from_zarr(args.zarr_file, yc=yc, xc=xc, patch_size=256, bands=bands)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model checkpoints
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Load config
    config = ckpt['config']
    
    # Load model
    model = load_model(config, ckpt, device)

    # Run inference
    run_inference(patch, model, device, ckpt['mean'], ckpt['std'], task=args.task, save_dir="inference_outputs")

    if args.task == "classification":
        # Save RGB for reference
        rgb = get_rgb(patch)
        plt.imsave(os.path.join("inference_outputs", "input_rgb_class_test.png"), rgb)