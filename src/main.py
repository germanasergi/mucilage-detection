import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from loguru import logger
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils.utils import load_config
from dataset.dataset import Sentinel2PatchDataset, Sentinel2NumpyDataset
from dataset.loader import define_loaders
from model_zoo.models import CNN, MILResNet, MILResNetMultiHead, build_timm_model
from torch.utils.data import DataLoader
from training.optim import EarlyStopping
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


def prepare_data(df_train, df_val, df_test, bands, batch_size=64, num_workers=4, res="r10m"):
    train_ds = Sentinel2NumpyDataset(df_train, bands, target_res=res, cache_file="saved_npy/train_cache.npz")
    val_ds   = Sentinel2NumpyDataset(df_val, bands, target_res=res, cache_file="saved_npy/val_cache.npz")
    test_ds  = Sentinel2NumpyDataset(df_test, bands, target_res=res, cache_file="saved_npy/test_cache.npz")

    # Normalize
    mean = np.nanmean(train_ds.X, axis=(0,1,2))
    std  = np.nanstd(train_ds.X, axis=(0,1,2))
    train_ds.X = (train_ds.X - mean) / std
    val_ds.X  = (val_ds.X - mean) / std
    test_ds.X = (test_ds.X - mean) / std


    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def diversity_loss(attn_maps):
    # attn_maps: [B, H, W]
    if isinstance(attn_maps, list):
        attn_maps = torch.stack(attn_maps, dim=0)
    B, Hn, H, W = attn_maps.shape
    attn_maps = attn_maps.view(B, Hn, -1)  # [B, num_heads, N]
    # Flatten
    attn_maps = attn_maps / (attn_maps.sum(dim=2, keepdim=True) + 1e-8)
    # Compute pairwise similarity between heads
    sims = []
    for i in range(Hn):
        for j in range(i + 1, Hn):
            sim = torch.sum(attn_maps[:, i, :] * attn_maps[:, j, :], dim=1)  # [B]
            sims.append(sim.mean())

    if len(sims) == 0:
        return torch.tensor(0.0, device=attn_maps.device)

    return torch.stack(sims).mean()



def build_model(config, num_classes):
    """
    Build a classification model (CNN or timm model).
    """
    in_channels = len(config['DATASET']['bands'])
    model_name = config['MODEL']['model_name']

    if model_name.startswith("MIL_"):
        base_name = model_name.replace("MIL_", "")
        logger.info(f"Using MIL model with backbone={base_name}, in_channels={in_channels}, num_classes={num_classes}")
        model = MILResNetMultiHead(
            model_name=base_name,
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=config['MODEL'].get('pretrained', True)
        )

    elif config['MODEL']['model_name'] == "CNN":
        logger.info(f"Using CNN with in_channels={in_channels}, num_classes={num_classes}")
        model = CNN(
            num_classes=num_classes,
            in_channels=in_channels,
            log_features=config['MODEL'].get('log_features', False)
        )

    else:
        logger.info(f"Using timm model {model_name} | in_channels={in_channels}, num_classes={num_classes}")
        pretrained = config['MODEL'].get('pretrained', True)
        model = build_timm_model(model_name, in_channels, num_classes, pretrained=pretrained)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), device


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
    if y is not None:
        class_counts = np.bincount(y)
        weights = 1.0 / class_counts
        weights = torch.tensor(weights, dtype=torch.float).to(device)

        if config['MODEL']['num_classes'] == 1:
            logger.info("Using weighted BCEWithLogitsLoss")
            criterion = nn.BCEWithLogitsLoss(pos_weight=weights[1])
        else:
            logger.info(f"Using weighted CrossEntropyLoss with weights={weights}")
            criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        if config['MODEL']['num_classes'] == 1:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

    return optimizer, criterion, scheduler_class

@track_emissions(save_to_api=True, experiment_id="91ba3396-1ad3-40a0-b64f-df074b7af5a7")
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        # outputs = model(inputs)
        # loss = criterion(outputs, labels)
        # ADDED FOR MULTIHEAD
        outputs, attn_maps = model(inputs)
        ce_loss = criterion(outputs, labels)
        loss = ce_loss + 0.05 * diversity_loss(attn_maps)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@track_emissions(save_to_api=True, experiment_id="91ba3396-1ad3-40a0-b64f-df074b7af5a7")
def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# def test_model(model, test_loader, criterion, device):
#     model.eval()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     y_true, y_prob, y_pred = [], [], []

#     with torch.no_grad():
#         for inputs, labels in tqdm(test_loader, desc="Testing", leave=False):
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)

#             running_loss += loss.item() * inputs.size(0)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#             probs = torch.softmax(outputs, dim=1)[:, 1]  # prob mucilage class
#             preds = (probs >= 0.7).long()

#             y_true.extend(labels.cpu().numpy())
#             y_prob.extend(probs.cpu().numpy())
#             y_pred.extend(preds.cpu().numpy())

#     epoch_loss = running_loss / total
#     epoch_acc = correct / total

#     results = pd.DataFrame({
#         "y_true": y_true,
#         "y_pred": y_pred,
#         "y_prob": y_prob
#     })

#     return epoch_loss, epoch_acc, results

@track_emissions(save_to_api=True, experiment_id="91ba3396-1ad3-40a0-b64f-df074b7af5a7")
def test_model(model, test_loader, criterion, device, save_attn_dir=None, attn_threshold=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true, y_prob, y_pred = [], [], []

    if save_attn_dir:
        os.makedirs(save_attn_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="Testing", leave=False)):
            inputs, labels = inputs.to(device), labels.to(device)

            # --- Handle models with attention maps ---
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                logits, attn_maps = outputs
            else:
                logits, attn_maps = outputs, None

            loss = criterion(logits, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            probs = torch.softmax(logits, dim=1)[:, 1]  # prob mucilage class
            preds = (probs >= 0.7).long()

            y_true.extend(labels.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            # --- Save attention maps if requested ---
            if attn_maps is not None and save_attn_dir:
                for b in range(inputs.size(0)):
                    if preds[b].item() == 1:  # only save mucilage predictions
                        patch_img = inputs[b].permute(1,2,0).cpu().numpy()
                        save_path = os.path.join(save_attn_dir, f"sample_{batch_idx}_{b}.png")
                        per_sample_maps = [head[b] for head in attn_maps]
                        save_multi_attention(per_sample_maps, patch_img, save_path=save_path)


    epoch_loss = running_loss / total
    epoch_acc = correct / total

    results = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob
    })

    return epoch_loss, epoch_acc, results


def main():

    parser = argparse.ArgumentParser(description='Classify Sentinel-2 patches')
    parser.add_argument('--patch_csv', type=str, required=True, help='Path to the patch CSV file')
    args = parser.parse_args()
    
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SRC_DIR)

    CFG_DIR = os.path.join(SRC_DIR, 'cfg')
    CSV_DIR = os.path.join(BASE_DIR, 'csv')

    config = load_config(config_path=os.path.join(CFG_DIR, 'config.yaml'))

    # Parameters from config
    bands = config['DATASET']['bands']
    num_epochs = config['TRAINING']['num_epochs']
    batch_size = config['TRAINING']['batch_size']
    res = config['DATASET']['res']
    pat = config['TRAINING']['patience']

    # Data 
    df = args.patch_csv  # top-left corners of patches + labels (optional)
    df_train, df_val, df_test = split_data(df)
    train_loader, val_loader, test_loader = prepare_data(df_train, df_val, df_test, bands, batch_size, res=res)

    # Labels
    if "label" in df_train.columns:
        y_train = df_train["label"].values
        num_classes = len(np.unique(y_train))
    else:
        y_train = None
        num_classes = config['MODEL'].get('num_classes', 2)  # fallback for inference

    # Model, optimizer, criterion
    model, device = build_model(config, num_classes=num_classes)
    optimizer, criterion, scheduler = build_opt(model, config, y=y_train, device=device)

    # Training loop
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

        # Early stopping check
        if early_stopping.step(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break


    # # Save metrics to CSV
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(os.path.join(SRC_DIR,"training_metrics_res_60.csv"), index=False)

    # Final test evaluation
    if "label" in df_test.columns:
        test_loss, test_acc, results = test_model(model, test_loader, criterion, device, save_attn_dir=os.path.join(SRC_DIR,"multi_attn_maps_bin_max"), attn_threshold=0.7)
        print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.3f}")

        results.to_csv(os.path.join(SRC_DIR, "test_predictions_res_60.csv"), index=False)
    else:
        print("No labels in test set → skipping evaluation.")

if __name__ == "__main__":
    main()