import torch.optim as optim
from torch.optim.lr_scheduler import StepLR  # or another scheduler
import torch.nn.functional as F
import torch

# Adam optimizer with weight decay
"""
optimizer = optim.Adam(
    model.parameters(),
    lr=float(LEARNING_RATE),
    weight_decay=0.0001  # L2 regularization factor
)
"""

# Learning rate scheduler
# Option 1: StepLR (reduces learning rate by gamma every step_size epochs)
"""
scheduler = StepLR(
    optimizer,
    step_size=30,  # decrease LR every 30 epochs
    gamma=0.1      # multiply LR by 0.1 at each step
)
"""

# Option 2: ReduceLROnPlateau (reduces learning rate when a metric plateaus)
"""
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',       # 'min' for loss, 'max' for accuracy
    factor=0.1,       # multiply LR by this factor
    patience=10,      # number of epochs with no improvement after which LR is reduced
    verbose=True      # print message when LR is reduced
)
"""

# Option 3: CosineAnnealingLR (cosine annealing schedule)
"""
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,  # maximum number of iterations
    eta_min=1e-6  # minimum learning rate
)
"""

def dice_loss_multiclass(logits, targets, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    # Only compute Dice on mucilage class (class 1)
    probs_fg = probs[:, 1, :, :]
    targets_fg = (targets == 1).float()
    intersection = (probs_fg * targets_fg).sum(dim=(1,2))
    union = probs_fg.sum(dim=(1,2)) + targets_fg.sum(dim=(1,2))
    dice = (2. * intersection + eps) / (union + eps)
    return 1 - dice.mean()

def combined_ce_dice_loss(logits, targets, ce_weight=0.5, class_weights=None):
    ce = F.cross_entropy(logits, targets.long(), weight=class_weights)
    dice = dice_loss_multiclass(logits, targets)
    return ce_weight * ce + (1 - ce_weight) * dice


class EarlyStopping:
    def __init__(self, patience=5, mode="min", min_delta=0.0):
        """
        Args:
            patience (int): how many epochs to wait without improvement before stopping
            mode (str): "min" (for loss) or "max" (for accuracy/metrics)
            min_delta (float): minimum change to qualify as an improvement
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, metric):
        if self.best is None:
            self.best = metric
            return False

        if self.mode == "min":
            improvement = self.best - metric
        else:  # mode == "max"
            improvement = metric - self.best

        if improvement > self.min_delta:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop