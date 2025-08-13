import os
import re
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from itertools import chain
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch_lr_finder import LRFinder

# Custom module imports
from model import PedestrianCrossingFusionNet
from model import LRWrapper
from DataLoader import get_data_loaders
from reconstruction.modules.utils.logger import Logger

# ----------------------------
# Hyperparameters
# ----------------------------
BATCH_SIZE = 4
EMBED_DIM = 16
N_EPOCHS = 30
CHECKPOINT_INTERVAL = 1
KERNEL_SIZE = 5
NUM_HEADS = 4
MAX_VOXEL_GRID_SIZE = int(1e2)
SPARSE_RATIO = 1.0
F_INPUT_DIM = 5
F_NUM_HEADS = 5
VOXEL_SIZE = 0.1
PATIENCE = 20
DROPOUT_RATE = 0.3

# ----------------------------
# Model Setup
# ----------------------------
# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = PedestrianCrossingFusionNet(
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    kernel_size=KERNEL_SIZE,
    max_voxel_grid_size=MAX_VOXEL_GRID_SIZE,
    sparse_ratio=SPARSE_RATIO,
    voxel_size=VOXEL_SIZE,
    feature_dim=F_INPUT_DIM,
    feature_num_heads=F_NUM_HEADS,
    dropout_rate=DROPOUT_RATE
).to(device)

# Collect all parameters
parameters = list(model.parameters())

# Initialize optimizer with weight decay
optimizer = torch.optim.AdamW(
    params=parameters,
    lr=1e-3
)

# ----------------------------
# resume training 
# ----------------------------
logger = Logger.get_logger(__name__)
if os.path.exists("checkpoints2"):
    logger.info("Checking for model checkpoints to resume training...")
    checkpoints = [f for f in os.listdir("checkpoints2/") if re.search(r"^checkpoint_epoch_\d+\.pth$", f)]
    sorted_checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if sorted_checkpoints:
        # Load the latest checkpoint
        latest_checkpoint = sorted_checkpoints[-1]
        logger.info(f"Loading checkpoint: {latest_checkpoint}")
        checkpoint = torch.load("checkpoints2/" + latest_checkpoint, map_location='cuda')

        # Load model state
        model.load_state_dict(checkpoint['model_state'])

        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        logger.info(f"Resumed from checkpoint: {latest_checkpoint}")
        # load epoch number to resume training
        start_epoch = checkpoint.get("epoch", 0)
        logger.info(f"Resuming training from epoch: {start_epoch + 1}")
    else:
        logger.warning("No valid epoch checkpoints found in 'checkpoints2/'. Starting training from scratch.")
# ----------------------------
# Config & Paths 
# ----------------------------
SCRIPT_PATH = Path(__file__).resolve().parent
LOKI_PATH = SCRIPT_PATH / 'loki_training_data'
LABEL_CSV_PATH = LOKI_PATH / "b_avatar_filtered_pedestrians.csv"  
PEDESTRIAN_DIR = LOKI_PATH / 'pedestrian_avatars'  
ENVIRONMENT_DIR = LOKI_PATH / '3d_constructed'
FEATURE_DIR = LOKI_PATH / 'pedestrian_features'  
CHECKPOINT_DIR = SCRIPT_PATH / "checkpoints2"
CHECKPOINT_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"
LOG_FILE = LOKI_PATH / 'train_log.log'
PLOT_PATH = LOKI_PATH / 'training_curves.png'

# ----------------------------
# Initialization
# ----------------------------
# Initialize Logger
Logger.configure_unified_file_logging(str(LOG_FILE))
logger = Logger.get_logger(__name__)
logger.info(f"Using device: {device}")

# Initialize data loaders with stratified split
train_dl, val_dl = get_data_loaders(
    PEDESTRIAN_DIR, ENVIRONMENT_DIR, FEATURE_DIR, LABEL_CSV_PATH,
    batch_size=BATCH_SIZE, train_set_percentage=0.7, val_set_percentage=0.3,
)

# Debug: Check a batch from train_dl
# first_batch = next(iter(train_dl))
# inputs, target = first_batch
# # Unpack inputs
# avatar, env, feats = inputs
# print("train_dl batch shapes:", avatar.shape, env.shape, feats.shape, target.shape)

# ----------------------------
# Class Weights
# ----------------------------
try:
    label_data = pd.read_csv(LABEL_CSV_PATH)
    class_counts = label_data['intended_actions'].value_counts()
    num_neg, num_pos = class_counts.get(0, 1), class_counts.get(1, 1)
    logger.info(f"Class counts: Negative: {num_neg}, Positive: {num_pos}")
    class_weight_value = num_neg / num_pos
    class_weights = torch.tensor([class_weight_value], dtype=torch.float).to(device)
    logger.info(f"Class weights computed: {class_weights.tolist()}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
except Exception as e:
    criterion = nn.BCEWithLogitsLoss()
    logger.warning(f"Warning: Could not compute class weights due to: {e}")

# ----------------------------
# Learning Rate Finder & Optimizer Adjustment
# ----------------------------
# lr_finder = LRFinder(LRWrapper(model), optimizer, criterion, device=device)
# lr_finder.range_test(train_dl, start_lr=1e-4, end_lr=1, num_iter=100)
# # Plot the learning rate finder results
# lr_finder.plot()
# # Find best learning rate based on steepest loss descent
# losses = np.array(lr_finder.history['loss'])
# lrs = np.array(lr_finder.history['lr'])
# best_lr = float(lrs[np.gradient(losses).argmin()]) / 5
# logger.info(f"Best learning rate found: {best_lr:.2E}")
# lr_finder.reset()

# Set the best learning rate for all parameter groups
# for param_group in optimizer.param_groups:
#     param_group['lr'] = best_lr

# Learning rate scheduler
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, T_max=N_EPOCHS, eta_min=best_lr * 0.01
# )

logger.info(f"Total trainable parameters: {sum(p.numel() for p in parameters if p.requires_grad):,}")

# ----------------------------
# Utility Functions
# ----------------------------
def compute_loss(outputs, targets):
    """Compute BCEWithLogitsLoss with proper reshaping"""
    return criterion(outputs.squeeze(), targets.float())

# ----------------------------
# Training & Validation Functions
# ----------------------------
def set_models_mode(mode):
    """Set all models to specified mode (train/eval)"""
    model.train() if mode == 'train' else model.eval()

def run_epoch(data_loader, is_train=True):
    """Run one epoch of training or validation"""
    set_models_mode('train' if is_train else 'eval')

    total_loss = 0.0
    total_samples = 0
    prediction_counts = {"crossing": 0, 'not_crossing': 0}
    target_counts = {"crossing": 0, 'not_crossing': 0}

    all_preds = []
    all_targets = []

    with torch.set_grad_enabled(is_train):
        for inputs, target in tqdm(data_loader, desc="Training" if is_train else "Validation", leave=False):
            avatar, env, feats = inputs
            avatar, env, feats, target = map(lambda x: x.to(device), (avatar, env, feats, target))

            outputs = model((avatar, env, feats))
            loss = compute_loss(outputs, target)

            acc_out = torch.sigmoid(outputs).squeeze()
            preds = (acc_out > 0.5).int()

            # Count predictions
            prediction_counts['crossing'] += int(preds.sum().item())
            prediction_counts['not_crossing'] += int((~preds.bool()).sum().item())
            # Count targets
            target_counts['crossing'] += (target == 1).sum().item()
            target_counts['not_crossing'] += (target == 0).sum().item()

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            # Accumulate for sklearn metrics
            all_preds.extend(acc_out.detach().cpu().numpy())
            all_targets.extend(target.detach().cpu().numpy())

            total_loss += loss.item() * target.size(0)
            total_samples += target.size(0)

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    bin_preds = (all_preds > 0.5).astype(int)

    # Compute metrics
    accuracy = accuracy_score(all_targets, bin_preds)
    f1 = f1_score(all_targets, bin_preds)
    weighted_f1 = f1_score(all_targets, bin_preds, average='weighted')
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = float('nan')  # Handle case with only one class present

    return total_loss / total_samples, accuracy, f1, auc, prediction_counts, target_counts, weighted_f1


# ----------------------------
# Main Training Loop
# ----------------------------

# Training state tracking
best_val_loss = float('inf')
no_improve_epochs = 0
train_losses, val_losses = [], []
train_accs, val_accs = [], []
total_epochs = start_epoch + N_EPOCHS if 'start_epoch' in locals() else N_EPOCHS
start_epoch = 0 if 'start_epoch' not in locals() else start_epoch

for epoch in range(N_EPOCHS):
    epoch += start_epoch  # Adjust epoch to resume from checkpoint
    logger.info(f"\nEpoch {epoch+1}/{total_epochs}")
    
    # Training phase
    train_loss, train_acc, trn_f1, trn_auc, train_cnts, tar_trn_cnts, trn_w_f1 = run_epoch(train_dl, is_train=True)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Validation phase
    val_loss, val_acc, vl_f1, vl_auc, val_cnts, tar_val_cnts, val_w_f1 = run_epoch(val_dl, is_train=False)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    # Update learning rate
    # scheduler.step()
    
    logger.info(
        f"\n"
        f"{'='*60}\n"
        f"Train:\n"
        f"  Loss         : {train_loss:.4f}\n"
        f"  Accuracy     : {train_acc:.2%}\n"
        f"  F1 Score     : {trn_f1:.2%}\n"
        f"  Weighted F1  : {trn_w_f1:.2%}\n"
        f"  AUC          : {trn_auc:.2%}\n"
        f"  Targets      : Crossing={tar_trn_cnts['crossing']}, Not Crossing={tar_trn_cnts['not_crossing']}\n"
        f"  Predictions  : Crossing={train_cnts['crossing']}, Not Crossing={train_cnts['not_crossing']}\n"
        f"Val:\n"
        f"  Loss         : {val_loss:.4f}\n"
        f"  Accuracy     : {val_acc:.2%}\n"
        f"  F1 Score     : {vl_f1:.2%}\n"
        f"  Weighted F1  : {val_w_f1:.2%}\n"
        f"  AUC          : {vl_auc:.2%}\n"
        f"  Targets      : Crossing={tar_val_cnts['crossing']}, Not Crossing={tar_val_cnts['not_crossing']}\n"
        f"  Predictions  : Crossing={val_cnts['crossing']}, Not Crossing={val_cnts['not_crossing']}\n"
        f"Learning Rate  : {optimizer.param_groups[0]['lr']:.2e}\n"
        f"{'='*60}"
    )

    # Checkpointing and early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_epochs = 0
        # Save complete model state
        torch.save({
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc
        }, BEST_MODEL_PATH)
        logger.info(f"Best model saved (val loss: {val_loss:.4f})")
    else:
        no_improve_epochs += 1
        logger.info(f"No improvement for {no_improve_epochs}/{PATIENCE} epochs")

    # Periodic checkpoint
    if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
        torch.save({
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc
        }, CHECKPOINT_DIR / f"checkpoint_epoch_{epoch+1}.pth")

    # Early stopping
    if no_improve_epochs >= PATIENCE:
        logger.info(f"Early stopping at epoch {epoch+1}")
        break

# ----------------------------
# Visualization
# ----------------------------
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(train_losses, 'o-', label='Train Loss')
plt.plot(val_losses, 'o-', label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(train_accs, 'o-', label='Train Accuracy')
plt.plot(val_accs, 'o-', label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.close()  
logger.info(f"Training curves saved to: {PLOT_PATH}")

# Final best model info
logger.info(f"\nTraining completed. Best validation loss: {best_val_loss:.4f}")
