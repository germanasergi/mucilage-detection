import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from model_zoo.models import CNN, MILResNet, MILResNetMultiHead, build_timm_model
from utils.plot import save_attention, save_multi_attention, get_rgb
import yaml
import timm


def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def load_model(config, ckpt, device):
    in_channels = len(config['DATASET']['bands'])
    base_name = config['MODEL']['model_name'].replace("MIL_", "")
    model = MILResNetMultiHead(
        model_name=base_name,
        in_channels=in_channels,
        num_classes=config['MODEL'].get('num_classes', 2),
        pretrained=False,  # no need for pretrained during inference
        num_heads=config['MODEL'].get('num_heads', 4)
    )
    
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    return model


def preprocess_patch(patch_array, mean, std):
    """
    Normalize patch using training statistics.
    patch_array: np.ndarray [H, W, C]
    """
    return (patch_array - mean) / (std + 1e-8)


def run_inference(patch_file, model, device, mean, std, save_dir="inference_outputs"):
    os.makedirs(save_dir, exist_ok=True)

    # --- Load patch (assume npy [H, W, C]) ---
    patch = np.load(patch_file)  # e.g. shape [H, W, C]
    print(patch['X'].shape)
    patch = preprocess_patch(patch['X'][0], mean, std)

    # --- Torch tensor ---
    inp = torch.tensor(patch, dtype=torch.float).permute(2, 0, 1).unsqueeze(0).to(device)  # [1, C, H, W]

    # --- Forward pass ---
    with torch.no_grad():
        logits, attn_maps = model(inp)
        probs = torch.softmax(logits, dim=1)[0, 1].item()
        pred = int(probs >= 0.7)

    print(f"Prediction: {pred} | Prob(mucilage)={probs:.3f}")

    # --- Save attention maps ---
    if pred == 1:  # only for mucilage
        patch_img = patch  # already normalized → convert back to RGB for display
        save_path = os.path.join(save_dir, os.path.basename(patch_file).replace(".npy", "_attn.png"))
        per_sample_maps = [head[0] for head in attn_maps]  # extract from batch
        save_multi_attention(per_sample_maps, patch_img, save_path=save_path)
        print(f"Saved attention maps → {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single patch")
    parser.add_argument("--patch_file", type=str, required=True, help="Path to input .npy patch file [H,W,C]")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model .pth")
    args = parser.parse_args()

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model checkpoints
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # --- Load config ---
    config = ckpt['config']
    
    # --- Load model ---
    model = load_model(config, ckpt, device)

    # --- Run inference ---
    run_inference(args.patch_file, model, device, ckpt['mean'], ckpt['std'])