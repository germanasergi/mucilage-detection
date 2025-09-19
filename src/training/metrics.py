import pandas as pd
import os
import matplotlib.pyplot as plt
import math
import numpy as np


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


def main():
    dir_path = os.getcwd()
    df = pd.read_csv(os.path.join(dir_path,"training_metrics_eff.csv"))
    test_numpy = np.load(os.path.join(dir_path,"saved_npy/test_cache.npz"), allow_pickle=True)['X']
    results = pd.read_csv(os.path.join(dir_path,"test_predictions_eff.csv"))

    plot_curve(df)
    plot_false_positives(test_numpy, results)

if __name__ == "__main__":
    main()