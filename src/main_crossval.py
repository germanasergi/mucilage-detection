import os
import argparse
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import yaml
from loguru import logger
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import auc, precision_recall_curve

from utils.utils import load_config
from dataset.dataset import Sentinel2PatchDataset, Sentinel2NumpyDataset
from dataset.loader import define_loaders, define_model, add_decoder_dropout
from model_zoo.models import CNN, MILResNet, MILResNetMultiHead, build_timm_model
from torch.utils.data import DataLoader
from training.optim import EarlyStopping, dice_loss_multiclass, combined_ce_dice_loss, FocalLoss, FocalLoss2d
from utils.plot import save_attention, save_multi_attention
from generate_dataset.generate_ds import setup_environment
#from training.metrics import MultiSpectralMetrics, avg_metric_bands

# Code carbon
from codecarbon import track_emissions


def split_data(labels_file, test_size=0.3, val_size=0.5, seed=6): # 42
    df = pd.read_csv(labels_file)

    # first split train vs test
    df_train, df_tmp = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    # then split train vs val
    df_test, df_val = train_test_split(
        df_tmp, test_size=val_size, stratify=df_tmp["label"], random_state=seed
    )

    df_trainval = pd.concat([df_train, df_val]).reset_index(drop=True)

    return df_trainval, df_test


def filter_mucilage_patches(df, mask_file):
    """
    Keep only patches whose mask contains mucilage pixels.
    Assumes same order between df and masks in mask_file.
    """
    masks = np.load(mask_file)["masks"]  # shape (N, H, W)
    keep_indices = [i for i, mask in enumerate(masks) if np.any(mask > 0)]
    return df.iloc[keep_indices].reset_index(drop=True)

def aggregate_train_val():
    
    # merge patches
    train_cache = np.load("saved_npy/train_cache_augmented.npz")["X"]
    val_cache   = np.load("saved_npy/val_cache.npz")["X"]
    trainval_cache = np.concatenate([train_cache, val_cache], axis=0)
    np.savez_compressed("saved_npy/trainval_cache.npz", X=trainval_cache)

    # merge masks
    train_masks = np.load("roboflow_dataset/saved_masks/train_masks_augmented.npz")["masks"]
    val_masks   = np.load("roboflow_dataset/saved_masks/val_masks.npz")["masks"]
    trainval_masks = np.concatenate([train_masks, val_masks], axis=0)
    np.savez_compressed("roboflow_dataset/saved_masks/trainval_masks.npz", masks=trainval_masks)


def prepare_data(df_trainval, df_test, train_idx, val_idx, bands, batch_size=64, num_workers=2, res="r10m", bbox=None, date=None, pat=None):

    df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
    df_val   = df_trainval.iloc[val_idx].reset_index(drop=True)

    train_ds = Sentinel2NumpyDataset(df_train, bands, target_res=res, cache_file="saved_npy/trainval_cache.npz", masks="roboflow_dataset/saved_masks/trainval_masks.npz", indices=train_idx, task="segmentation", transform=False, bbox=bbox, date=date, pat=pat)
    val_ds   = Sentinel2NumpyDataset(df_val, bands, target_res=res, cache_file="saved_npy/trainval_cache.npz", masks="roboflow_dataset/saved_masks/trainval_masks.npz", indices=val_idx, task="segmentation", transform=False, bbox=bbox, date=date, pat=pat)
    test_ds  = Sentinel2NumpyDataset(df_test, bands, target_res=res, cache_file="saved_npy/test_cache.npz", masks="roboflow_dataset/saved_masks/test_masks.npz", indices=None, task="segmentation", transform=False, bbox=bbox, date=date, pat=pat)

    # Normalize
    mean = np.nanmean(train_ds.X, axis=(0,1,2))
    std  = np.nanstd(train_ds.X, axis=(0,1,2))
    train_ds.X = (train_ds.X - mean) / std
    val_ds.X   = (val_ds.X - mean) / std
    test_ds.X  = (test_ds.X - mean) / std

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, mean, std


def build_model(config):

    model = define_model(
        name=config['MODEL']['model_name'],
        encoder_name=config['MODEL']['encoder_name'],
        encoder_weights = None,
        in_channel=len(config['DATASET']['bands']),
        out_channels=config['MODEL']['num_classes'],
        activation=config['MODEL']['activation'],
        dropout=False
        )
    add_decoder_dropout(model, p=0.3)

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

    if config['MODEL']['num_classes'] == 2:
        logger.info("Using weighted BCEWithLogitsLoss")
        criterion = FocalLoss2d(alpha=0.95, gamma=2) #nn.BCEWithLogitsLoss(pos_weight=weights[1])
        print("Using Focal Loss for binary segmentation")
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
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        # ---- Compute pixel accuracy ----
        with torch.no_grad():
            if outputs.shape[1] == 1:  # Binary segmentation
                preds = (torch.sigmoid(outputs) > 0.8).float()
                acc = (preds == masks).float().mean().item()
            else:  # Multi-class segmentation
                preds = (torch.softmax(outputs, dim=1)[:,1] > 0.5).float()
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
def evaluate_model(model, val_loader, device):
    model.eval()
    y_true_all, y_prob_all = [], []

    with torch.no_grad():
        for inputs, masks in tqdm(val_loader, desc="Validation", leave=False):
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if outputs.shape[1] == 1:  # binary
                probs = torch.sigmoid(outputs)
            else:  # multi-class
                probs = torch.softmax(outputs, dim=1)[:,1,:,:]

            y_true_all.append(masks.cpu().numpy().ravel())
            y_prob_all.append(probs.cpu().detach().numpy().ravel())

    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_prob_all)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    return pr_auc


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
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # Binary segmentation
            if outputs.shape[1] == 1:
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.8).float().squeeze(1)
            else:
                probs = torch.softmax(outputs, dim=1)[:,1,:,:]
                preds = (probs > 0.8).float()
                if masks.ndim == 4:
                    masks = masks.squeeze(1)

            # Compute loss
            if criterion is not None:
                loss = criterion(outputs, masks)
                running_loss += loss.item() * inputs.size(0)

            # Pixel accuracy
            acc = (preds == masks).float().mean().item()
            running_acc += acc * inputs.size(0)
            total += inputs.size(0)

            y_true_all.append(masks.cpu().numpy().ravel())
            y_prob_all.append(probs.cpu().detach().numpy().ravel())
            y_pred_all.append(preds.cpu().numpy().ravel())

            # Save predicted masks if requested
            if save_preds_dir:
                for b in range(inputs.size(0)):
                    pred_np = preds[b].squeeze()
                    save_path = os.path.join(save_preds_dir, f"pred_{i}_{b}.png")
                    cv2.imwrite(save_path, (pred_np * 255).astype(np.uint8))

    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_prob_all)
    y_pred = np.concatenate(y_pred_all)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    epoch_loss = running_loss / total if criterion is not None else None
    epoch_acc = running_acc / total

    results = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob
    })

    return epoch_loss, epoch_acc, pr_auc, results



def main():
    parser = argparse.ArgumentParser(description='Segmentation of Sentinel-2 patches')
    parser.add_argument('--patch_csv', type=str, required=True, help='Path to the patch CSV file')
    args = parser.parse_args()
    
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SRC_DIR)
    CFG_DIR = os.path.join(SRC_DIR, 'cfg')

    # --- Load config ---
    config = load_config(config_path=os.path.join(CFG_DIR, 'config.yaml'))
    config_dataset = load_config(config_path=os.path.join(CFG_DIR, 'config_dataset.yaml'))
    env = setup_environment(config_dataset)

    # --- Parameters from config ---
    bands = config['DATASET']['bands']
    num_epochs = config['TRAINING']['num_epochs']
    batch_size = config['TRAINING']['batch_size']
    res = config['DATASET']['res']
    patience = config['TRAINING']['patience']
    bbox = config_dataset['query']['bbox']
    pat = env['PAT']
    start_date = datetime.strptime(config_dataset['query']['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime(config_dataset['query']['end_date'], '%Y-%m-%d')
    mid_date = start_date + (end_date - start_date) / 2

    # Core change for cross-validation
    if not os.path.exists("saved_npy/trainval_cache.npz") or not os.path.exists("roboflow_dataset/saved_masks/trainval_masks.npz"):
        aggregate_train_val()

    # --- Data ---
    df_trainval, df_test = split_data(args.patch_csv)

    if torch.cuda.is_available():
        logger.info("Using GPU for training")

    # --- K-FOLD SETUP ---
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=6)  # 42

    best_overall_pr_auc = -np.inf
    best_model_state = None
    best_fold = None
    best_mean = None
    best_std = None

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_trainval, df_trainval["label"])):
        print(f"\n========== FOLD {fold+1} ==========")

        # --- Prepare fold data loaders ---
        train_loader, val_loader, test_loader, mean, std = prepare_data(
            df_trainval=df_trainval,
            df_test=df_test,
            train_idx=train_idx,
            val_idx=val_idx,
            bands=bands,
            batch_size=batch_size,
            res=res,
            bbox=bbox,
            date=mid_date,
            pat=pat
        )

        # ---- collect masks for weighted loss ----
        all_masks = []
        for _, mask in train_loader.dataset:
            all_masks.append(mask.cpu().numpy().ravel())
        y = np.concatenate(all_masks)

        # --- Build model, optimizer, criterion ---
        model, device = build_model(config)
        optimizer, criterion, scheduler = build_opt(model, config, y, device=device)
        early_stopping = EarlyStopping(patience=patience, mode="min")

        best_val_pr_auc = -np.inf

        # --- TRAINING LOOP ---
        for epoch in range(num_epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_pr_auc = evaluate_model(model, val_loader, device)

            print(f"Fold {fold+1} | Epoch {epoch+1}/{num_epochs} "
                f"| Train loss: {train_loss:.4f}, acc: {train_acc:.3f} "
                f"| Val PR-AUC: {val_pr_auc:.4f}")

            if scheduler:
                scheduler.step(-val_pr_auc)  # maximize PR-AUC, so use negative if using ReduceLROnPlateau

            if early_stopping.step(-val_pr_auc):
                print("Early stopping triggered")
                break

            best_val_pr_auc = max(best_val_pr_auc, val_pr_auc)
            torch.cuda.empty_cache()
            gc.collect()

        print(f"Fold {fold+1} | Best val PR-AUC: {best_val_pr_auc:.4f}")

        # --- Track best model across folds ---
        if best_val_pr_auc > best_overall_pr_auc:
            best_overall_pr_auc = best_val_pr_auc
            best_model_state = model.state_dict()
            best_fold = fold + 1
            best_mean = mean
            best_std = std

        # Optional: test evaluation for this fold
        test_loss, test_acc, test_pr_auc, _ = test_model(model, test_loader, criterion, device)
        print(f"Fold {fold+1} | Test PR-AUC: {test_pr_auc:.4f} | Test pixel accuracy: {test_acc:.4f}")

        fold_results.append({
            "fold": fold + 1,
            "best_val_pr_auc": best_val_pr_auc,
            "test_pr_auc": test_pr_auc,
            "test_acc": test_acc
        })

        del (y, all_masks, model, optimizer, criterion, train_loader, val_loader, test_loader)
        torch.cuda.empty_cache()
        gc.collect()

    # --- SAVE BEST MODEL ---
    checkpoint_path = os.path.join(SRC_DIR, "training/unet_best_kfold.pth")
    torch.save({
        "model_state": best_model_state,
        "mean": best_mean,
        "std": best_std,
        "config": config
    }, checkpoint_path)
    print(f"\nBest model saved from fold {best_fold} to {checkpoint_path} | PR-AUC: {best_overall_pr_auc:.4f}")

    # --- SAVE K-FOLD RESULTS ---
    df_results = pd.DataFrame(fold_results)
    df_results.to_csv(os.path.join(SRC_DIR, "kfold_results_unet.csv"), index=False)

    print("\nK-Fold Summary:")
    print(df_results)
    print("Mean test PR-AUC:", df_results["test_pr_auc"].mean())
    print("Std test PR-AUC:", df_results["test_pr_auc"].std())

if __name__ == "__main__":
    main()