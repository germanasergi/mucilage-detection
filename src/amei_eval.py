import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from tqdm import tqdm
from dataset.patches import *
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

def compute_amei(ds, eps=1e-6):
    red   = ds["measurements/reflectance/r10m/b04"] / 10000.0
    green = ds["measurements/reflectance/r10m/b03"] / 10000.0
    nir   = resample_to_10m(ds, 'b8a', 'b04', folder='measurements') / 10000.0
    swir  = resample_to_10m(ds, 'b11', 'b04', folder='measurements') / 10000.0

    # AMEI = (2*red + nir - 2*swir) / (green + 0.25*swir)
    denom = green + 0.25 * swir
    amei  = (2*red + nir - 2*swir) / (denom + eps)

    return amei.rename("amei") 

# df_patches contains columns: ['zarr_path', 'x', 'y', 'label']
patch_size = 256
patches_csv = "/home/ubuntu/mucilage_pipeline/mucilage-detection/csv/patches_final.csv"  # CSV with columns: ['zarr_path', 'x', 'y', 'label']
df_patches = pd.read_csv(patches_csv)


y_true = []
y_score = []  # continuous (max AMEI in patch)

for _, row in tqdm(df_patches.iterrows(), total=len(df_patches)):
    zarr_path = row['zarr_path']
    x, y = row['x'], row['y']
    label = row['label']

    ds = xr.open_datatree(zarr_path, engine="zarr", mask_and_scale=False, chunks={})
    amei_map = compute_amei(ds)
    patch_amei = amei_map[y:y+patch_size, x:x+patch_size]
    ds.close()

    mac = np.nanmax(patch_amei)
    y_true.append(label)
    y_score.append(mac)

y_true = np.array(y_true)
y_score = np.array(y_score)

np.save("y_true.npy", y_true)
np.save("y_score_amei.npy", y_score)

# y_true = np.load("y_true.npy")
# y_score = np.load("y_score_amei.npy")

# sweep thresholds for F1 
thresholds = np.linspace(0, 1, 10)
f1_scores = []

for thresh in thresholds:
    y_pred = (y_score >= thresh).astype(int)
    f1 = f1_score(y_true, y_pred)
    f1_scores.append(f1)

# Plot F1 vs threshold
plt.figure()
plt.plot(thresholds, f1_scores, marker='o')
plt.xlabel("AMEI threshold")
plt.ylabel("F1 score")
plt.title("Patch-level F1 vs AMEI threshold")
plt.grid(True)
plt.show()
plt.savefig("f1_vs_threshold.png", dpi=300)
plt.close()

# ROC & AUC
fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
auc_value = roc_auc_score(y_true, y_score)

plt.figure()
plt.plot(fpr, tpr, label=f"AMEI (AUC={auc_value:.3f})")
plt.plot([0,1], [0,1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - AMEI")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("roc_curve.png", dpi=300)
plt.close()

print(f"Best F1: {max(f1_scores):.3f} at threshold={thresholds[np.argmax(f1_scores)]:.3f}")
print(f"ROC AUC for AMEI: {auc_value:.3f}")