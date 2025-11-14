import pandas as pd
import os
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve, f1_score, precision_score, recall_score, jaccard_score


def plot_curve(df):
    # Loss
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Accuracy
    plt.subplot(1,2,2)
    plt.plot(df["epoch"], df["train_acc"], label="Train Acc")
    plt.plot(df["epoch"], df["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_loss_curve.png")

    plt.tight_layout()
    plt.show()



def plot_false_positives(test_numpy, results, save_dir="plots", ncols=5):
    os.makedirs(save_dir, exist_ok=True)

    # Load indices
    y_true = results["y_true"].values
    y_pred = results["y_pred"].values
    false_pos_idx = np.where((y_true == 0) & (y_pred == 1))[0]
    n_patches = len(false_pos_idx)
    nrows = math.ceil(n_patches / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(2*ncols, 2*nrows))
    axes = np.array(axes).reshape(-1)

    for ax, i in zip(axes, false_pos_idx):
        patch = test_numpy[i]

        # Simple RGB normalization
        rgb = patch[:, :, [3, 2, 1]] #now there is b1 as well
        p2, p98 = np.nanpercentile(rgb, (2, 98))
        rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)

        ax.imshow(rgb)
        ax.set_title(f"Idx {i}")
        ax.axis("off")
        
    for ax in axes[len(false_pos_idx):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "false_positives.png"))
    plt.show()


def plot_segmentation_false_positives(
    test_patches,        # numpy array (N, H, W, C)
    results,             # numpy array (N, H, W)
    save_dir="plots",
    ncols=3,
    highlight_color=(1, 0, 0, 0.5)  # red overlay for FPs
    ):
    """
    Visualize segmentation false positives (pred=1, gt=0).
    Highlights false positive pixels in red over the RGB image.
    """
    os.makedirs(save_dir, exist_ok=True)
    pred_masks = (results["y_pred"].values.reshape(-1, 256, 256)).astype(int)
    gt_masks = (results["y_true"].values.reshape(-1, 256, 256)).astype(int)

    # Find patches that contain at least one false positive pixel
    fp_idx = [
        i for i in range(len(gt_masks))
        if np.any((pred_masks[i] == 1) & (gt_masks[i] == 0))
    ]
    if not fp_idx:
        print("No false positives found.")
        return
    
    nrows = len(fp_idx[:30])
    print(f"Found {len(fp_idx)} patches with false positives")
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3))
    if nrows == 1:
        axes = np.expand_dims(axes, 0)  # handle single case

    for r, i in enumerate(fp_idx[:30]):  # Show up to 15 examples
        patch = test_patches[i]
        gt = gt_masks[i]
        pred = pred_masks[i]

        # Normalize RGB
        rgb = patch[:, :, [3, 2, 1]]
        p2, p98 = np.nanpercentile(rgb, (2, 98))
        rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)

        # Create color overlay for errors
        overlay = np.zeros_like(rgb)
        overlay[(pred == 1) & (gt == 1)] = [0, 1, 0]   # True Positive → Green
        overlay[(pred == 1) & (gt == 0)] = [1, 0, 0]   # False Positive → Red
        overlay[(pred == 0) & (gt == 1)] = [0, 0, 1]   # False Negative → Blue

        # Panels
        axes[r, 0].imshow(rgb)
        axes[r, 0].set_title(f"RGB Patch {i}")
        axes[r, 0].axis("off")

        axes[r, 1].imshow(gt, cmap="gray")
        axes[r, 1].set_title("Ground Truth")
        axes[r, 1].axis("off")

        axes[r, 2].imshow(rgb)
        axes[r, 2].imshow(overlay, alpha=0.5)
        axes[r, 2].set_title("Prediction Overlay")
        axes[r, 2].axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "fp_triplets.png")
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"Saved visualization to {save_path}")


def plot_auc(results):
    y_true = results["y_true"].values
    y_score = results["y_prob"].values
    y_pred = results["y_pred"].values

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    auc_value = roc_auc_score(y_true, y_score)

    plt.figure()
    plt.plot(fpr, tpr, label=f"U-net (AUC={auc_value:.3f})")
    plt.plot([0,1], [0,1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - U-net")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("roc_curve.png", dpi=300)
    plt.close()

    print(f"ROC AUC: {auc_value:.3f}")

def print_metrics(results):
    y_true = results["y_true"].values
    y_score = results["y_prob"].values
    y_pred = results["y_pred"].values

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)

    print(f"Precision: {precision_score(y_true, y_pred)}, Recall: {recall_score(y_true, y_pred)}")
    print(f"F1 Score: {f1_score(y_true, y_pred)}")
    print(f"IoU: {jaccard_score(y_true, y_pred)}")
    print(f"Precision-Recall AUC: {pr_auc:.4f}")


def main():
    dir_path = os.getcwd()
    df = pd.read_csv(os.path.join(dir_path,"training_results/training_metrics_unet_noweights.csv"))
    test_numpy = np.load(os.path.join(dir_path,"saved_npy/test_cache.npz"), allow_pickle=True)['X']
    results = pd.read_csv(os.path.join(dir_path,"training_results/test_predictions_unet_noweights.csv"))

    #plot_curve(df)
    # plot_false_positives(test_numpy, results)
    # plot_segmentation_false_positives(test_patches=test_numpy, results=results)
    # plot_auc(results)
    print_metrics(results)

if __name__ == "__main__":
    main()