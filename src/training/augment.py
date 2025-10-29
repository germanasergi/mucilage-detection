import numpy as np
import os
import random
import cv2
from tqdm import tqdm
import albumentations as A

# Input
patches = np.load("/home/ubuntu/mucilage_pipeline/mucilage-detection/saved_npy/train_cache_sst.npz")["X"]
masks = np.load("/home/ubuntu/mucilage_pipeline/mucilage-detection/roboflow_dataset/saved_masks/train_masks.npz")["masks"]

print("Original:", patches.shape, masks.shape)

# Define augmentations (tuned for satellite patches)
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5)  # you can change to 128 if your patch size is larger
])

# Create lists to store augmented data
augmented_patches = []
augmented_masks = []

# Loop through all patches
for patch, mask in tqdm(zip(patches, masks), total=len(patches)):
    # Only augment mucilage patches
    if mask.sum() > 0:
        # Generate 2–3 augmentations per patch (adjust as needed)
        for _ in range(3):
            transformed = augment(image=patch, mask=mask)
            augmented_patches.append(transformed["image"])
            augmented_masks.append(transformed["mask"])

# Combine with original data
augmented_patches = np.stack(augmented_patches)
augmented_masks = np.stack(augmented_masks)

X_aug = np.concatenate([patches, augmented_patches])
M_aug = np.concatenate([masks, augmented_masks])

print("Augmented dataset:", X_aug.shape, M_aug.shape)

# Count mucilage patches before and after augmentation
original_mucilage_count = np.sum([1 for m in masks if m.sum() > 0])
augmented_mucilage_count = np.sum([1 for m in M_aug if m.sum() > 0])
print(f"Original mucilage patches: {original_mucilage_count}")
print(f"Augmented mucilage patches: {augmented_mucilage_count}")

# Save
os.makedirs("saved_npy", exist_ok=True)
np.savez_compressed("/home/ubuntu/mucilage_pipeline/mucilage-detection/saved_npy/train_cache_sst_augmented.npz", X=X_aug)
np.savez_compressed("/home/ubuntu/mucilage_pipeline/mucilage-detection/roboflow_dataset/saved_masks/train_masks_sst_augmented.npz", masks=M_aug)