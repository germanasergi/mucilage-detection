import os
import argparse
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from loguru import logger
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils.utils import load_config
from dataset.dataset import Sentinel2PatchDataset, Sentinel2NumpyDataset
from dataset.loader import define_loaders, define_model
from model_zoo.models import CNN, MILResNet, MILResNetMultiHead, build_timm_model
from torch.utils.data import DataLoader
from training.optim import EarlyStopping, dice_loss_multiclass, combined_ce_dice_loss
from utils.plot import save_attention, save_multi_attention
#from training.metrics import MultiSpectralMetrics, avg_metric_bands

# Code carbon
from codecarbon import track_emissions


def split_data(labels_file, test_size=0.3, val_size=0.5, seed=42):
    df = pd.read_csv(labels_file)

    # first split train vs test
    df_train, df_tmp = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    # then split train vs val
    df_test, df_val = train_test_split(
        df_tmp, test_size=val_size, stratify=df_tmp["label"], random_state=seed
    )
    return df_train, df_val, df_test


def prepare_data(df_train, df_val, df_test, bands, batch_size=64, num_workers=2, res="r10m"):
    train_ds = Sentinel2NumpyDataset(df_train, bands, target_res=res, cache_file="saved_npy/train_cache.npz", masks="saved_npy/train_masks_refined.npz", task="segmentation")
    val_ds   = Sentinel2NumpyDataset(df_val, bands, target_res=res, cache_file="saved_npy/val_cache.npz", masks="saved_npy/val_masks_refined.npz", task="segmentation")
    test_ds  = Sentinel2NumpyDataset(df_test, bands, target_res=res, cache_file="saved_npy/test_cache.npz", masks="saved_npy/test_masks_refined.npz", task="segmentation")

    # Normalize
    mean = np.nanmean(train_ds.X, axis=(0,1,2))
    std  = np.nanstd(train_ds.X, axis=(0,1,2))
    train_ds.X = (train_ds.X - mean) / std
    val_ds.X  = (val_ds.X - mean) / std
    test_ds.X = (test_ds.X - mean) / std


    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, mean, std


def build_model(config):


    model = define_model(
        name=config['MODEL']['model_name'],
        encoder_name=config['MODEL']['encoder_name'],
        encoder_weights = config['MODEL']['encoder_weights'],
        in_channel=len(config['DATASET']['bands']),
        out_channels=config['MODEL']['num_classes'],
        activation=config['MODEL']['activation'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model, device



def build_opt(model, config, y=None, device="cpu"):
    """
    Build optimizer, criterion and scheduler.
    - Supports weighted loss if labels `y` are provided.
    """
    # ----- Optimizer -----
    optimizer_class = getattr(optim, config['TRAINING']['optim'])
    optimizer = optimizer_class(
        model.parameters(),
        lr=float(config['TRAINING']['learning_rate'])
    )

    # ----- Scheduler -----
    scheduler_class = None
    if config['TRAINING'].get('scheduler', False):
        lr_scheduler = getattr(optim.lr_scheduler, config['TRAINING']['scheduler_type'])
        scheduler_class = lr_scheduler(
            optimizer,
            mode='min',
            factor=config['TRAINING']['factor'],
            patience=config['TRAINING'].get('patience', 5)
        )
        logger.info(f"Scheduler: {config['TRAINING']['scheduler_type']}")

    # ----- Loss function -----
    class_counts = np.bincount(y)
    weights = 1.0 / class_counts
    weights = torch.tensor(weights, dtype=torch.float).to(device)

    if config['MODEL']['num_classes'] == 1:
        logger.info("Using weighted BCEWithLogitsLoss")
        criterion = nn.BCEWithLogitsLoss(pos_weight=weights[1])
    else:
        logger.info(f"Using weighted CrossEntropyLoss with weights={weights}")
        #criterion = nn.CrossEntropyLoss(weight=weights)
        print("Using combined CE + Dice loss")
        criterion = lambda logits, masks: combined_ce_dice_loss(
        logits, masks, ce_weight=0.5, class_weights=weights
        )

    return optimizer, criterion, scheduler_class

@track_emissions(save_to_api=True, experiment_id="91ba3396-1ad3-40a0-b64f-df074b7af5a7")
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss, running_acc, total = 0.0, 0.0, 0

    for inputs, masks in tqdm(dataloader, desc="Training", leave=False):
        inputs, masks = inputs.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        # ---- Compute pixel accuracy ----
        with torch.no_grad():
            if outputs.shape[1] == 1:  # Binary segmentation
                preds = (torch.sigmoid(outputs) > 0.5).float()
                acc = (preds == masks).float().mean().item()
            else:  # Multi-class segmentation
                preds = torch.argmax(outputs, dim=1)
                if masks.ndim == 4:  # (B,1,H,W)
                    masks = masks.squeeze(1)
                acc = (preds == masks).float().mean().item()

        running_loss += loss.item() * inputs.size(0)
        running_acc += acc * inputs.size(0)
        total += inputs.size(0)

        del inputs, masks, outputs, loss, preds
        torch.cuda.empty_cache()
        gc.collect()

    epoch_loss = running_loss / total
    epoch_acc = running_acc / total
    return epoch_loss, epoch_acc


@track_emissions(save_to_api=True, experiment_id="91ba3396-1ad3-40a0-b64f-df074b7af5a7")
def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss, running_acc, total = 0.0, 0.0, 0

    with torch.no_grad():
        for inputs, masks in tqdm(val_loader, desc="Validation", leave=False):
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, masks)

            # ---- Pixel accuracy ----
            if outputs.shape[1] == 1:  # Binary
                preds = (torch.sigmoid(outputs) > 0.5).float()
                acc = (preds == masks).float().mean().item()
            else:  # Multi-class
                preds = torch.argmax(outputs, dim=1)
                if masks.ndim == 4:
                    masks = masks.squeeze(1)
                acc = (preds == masks).float().mean().item()

            running_loss += loss.item() * inputs.size(0)
            running_acc += acc * inputs.size(0)
            total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_acc / total
    return epoch_loss, epoch_acc


@track_emissions(save_to_api=True, experiment_id="91ba3396-1ad3-40a0-b64f-df074b7af5a7")
def test_model(model, test_loader, criterion, device, save_preds_dir=None):
    model.eval()
    running_loss, running_acc, total = 0.0, 0.0, 0
    y_true_all, y_prob_all, y_pred_all = [], [], []

    if save_preds_dir:
        os.makedirs(save_preds_dir, exist_ok=True)

    with torch.no_grad():
        for i, (inputs, masks) in enumerate(tqdm(test_loader, desc="Testing", leave=False)):
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, masks)

            # Handle binary vs multi-class
            if outputs.shape[1] == 1:  # Binary segmentation
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float().squeeze(1)
                acc = (preds == masks).float().mean().item()
            else:
                probs = torch.softmax(outputs, dim=1)[:, 1, :, :]  # shape [B, H, W]
                preds = (probs > 0.5).float()
                if masks.ndim == 4:
                    masks = masks.squeeze(1)
                acc = (preds == masks).float().mean().item()

            running_loss += loss.item() * inputs.size(0)
            running_acc += acc * inputs.size(0)
            total += inputs.size(0)

            y_true_all.append(masks.detach().cpu().numpy().ravel())
            y_prob_all.append(probs.detach().cpu().numpy().ravel())
            y_pred_all.append(preds.detach().cpu().numpy().ravel())

            # Optionally save some predictions as .png
            if save_preds_dir:
                for b in range(inputs.size(0)):
                    pred_np = preds[b].squeeze().cpu().numpy()
                    save_path = os.path.join(save_preds_dir, f"pred_{i}_{b}.png")
                    cv2.imwrite(save_path, (pred_np * 255).astype(np.uint8))

    epoch_loss = running_loss / total
    epoch_acc = running_acc / total

    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_prob_all)
    y_pred = np.concatenate(y_pred_all)

    results = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob
    })

    return epoch_loss, epoch_acc, results



def main():
    parser = argparse.ArgumentParser(description='Segmentation of Sentinel-2 patches')
    parser.add_argument('--patch_csv', type=str, required=True, help='Path to the patch CSV file')
    args = parser.parse_args()
    
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SRC_DIR)
    CFG_DIR = os.path.join(SRC_DIR, 'cfg')

    # --- Load config ---
    config = load_config(config_path=os.path.join(CFG_DIR, 'config.yaml'))

    # --- Parameters from config ---
    bands = config['DATASET']['bands']
    num_epochs = config['TRAINING']['num_epochs']
    batch_size = config['TRAINING']['batch_size']
    res = config['DATASET']['res']
    pat = config['TRAINING']['patience']

    # --- Data ---
    df_train, df_val, df_test = split_data(args.patch_csv)
    train_loader, val_loader, test_loader, mean, std = prepare_data(df_train, df_val, df_test, bands, batch_size, res=res)
    all_masks = []
    for _, mask in train_loader.dataset:
        all_masks.append(mask.cpu().numpy().astype(int).ravel())
    y = np.concatenate(all_masks)

    # --- Model, optimizer, criterion ---
    model, device = build_model(config)
    optimizer, criterion, scheduler = build_opt(model, config, y, device=device)

    # --- Training history ---
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    early_stopping = EarlyStopping(patience=pat, mode="min")

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs} "
              f"| Train loss: {train_loss:.4f}, acc: {train_acc:.3f} "
              f"| Val loss: {val_loss:.4f}, acc: {val_acc:.3f}")

        history["epoch"].append(epoch+1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if scheduler:
            scheduler.step(val_loss)

        if early_stopping.step(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break

        # Save metrics to CSV
        df_hist = pd.DataFrame(history)
        df_hist.to_csv(os.path.join(SRC_DIR,"training_metrics_unet.csv"), index=False)

        del train_loss, train_acc, val_loss, val_acc
        torch.cuda.empty_cache()
        gc.collect()

    # Save model checkpoint
    checkpoint_path = os.path.join(SRC_DIR, "training/unet_checkpoint.pth")
    torch.save({
        "model_state": model.state_dict(),
        "mean": mean,
        "std": std,
        "config": config
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    # Final test evaluation
    print("\n Running final test evaluation...")
    test_loss, test_acc, results = test_model(
        model,
        test_loader,
        criterion,
        device,
        save_preds_dir=None
    )

    print(f"Final Test Loss: {test_loss:.4f} | Test Pixel Accuracy: {test_acc:.4f}")
    results.to_csv(os.path.join(SRC_DIR, "test_predictions_unet.csv"), index=False)

if __name__ == "__main__":
    main()