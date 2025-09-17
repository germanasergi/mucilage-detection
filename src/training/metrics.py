import pandas as pd
import os
import matplotlib.pyplot as plt

dir_path = os.getcwd()
df = pd.read_csv(os.path.join(dir_path,"training_metrics_allbands_10.csv"))
plt.figure(figsize=(12,5))

# Loss
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