import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from itertools import chain
# from torch_snippets.torch_loader import Report
from linear_fusion_head.attention_Base_fusion import AttentionFusionHead
from attention_vector.point_cloud_attn_vector import PointCloudAttentionModel
from attention_vector.Pedestrian_PC_Attention.Ped_Att_Model import PointNetFeatureExtractor
from features_extraction.Multi_head_attention_layer import MultiHeadAttention

from DataLoader import get_data_loaders
from tqdm import tqdm

# Define the paths
SCRIPT_PATH = Path(__file__).resolve().parent
SCRIPT_PARENT_PATH = SCRIPT_PATH.parent
LOKI_PATH = SCRIPT_PARENT_PATH / 'loki_training_data'
LABEL_CSV_PATH = LOKI_PATH / "b_avatar_filtered_pedistrians.csv"
PEDESTRIAN_DIR = LOKI_PATH / 'pedistrian_avatars'
ENVIRONMENT_DIR = LOKI_PATH / '3d_constructed'
FEATURE_DIR = LOKI_PATH / 'pedistrian_featuers'

# Global params
BATCH_SIZE = 8
EMBED_DIM = 32
N_EPOCHS = 2
CHECKPOINT_INTERVAL = 1 #Save model every 5 epochs

# Model save paths
# BEST_MODEL_PATH = "best_model.pth"
# FINAL_MODEL_PATH = "final_model.pth"
# CHECKPOINT_PATH = "checkpoint_epoch_{epoch}.pth"

# Params for model (CAV & LWSA)
KERNEL_SIZE = 3
NUM_HEADS = 4
MAX_VOXEL_GRID_SIZE = int(1e2)
SPARSE_RATIO = 0.5
device = 'cpu' 

# Params for model (multi head attention model)
f_input_dim = 5
f_num_heads = 5

# Create data loaders
train_dl, val_dl, test_dl = get_data_loaders(
    PEDESTRIAN_DIR, ENVIRONMENT_DIR, FEATURE_DIR, LABEL_CSV_PATH,
    batch_size=BATCH_SIZE, train_set_percentage=0.7, val_set_percentage=0.15
)

# init attention model (CAV & LWSA)
pcd_attention_model = PointCloudAttentionModel(
    embed_dim=EMBED_DIM, 
    kernel_size=KERNEL_SIZE,
    num_heads=NUM_HEADS, 
    max_voxel_grid_size=MAX_VOXEL_GRID_SIZE, 
    sparse_ratio=SPARSE_RATIO
).to(device) 

pointnet_model = PointNetFeatureExtractor(input_dim=3, output_dim=EMBED_DIM).to(device) 

featuers_attention_model = MultiHeadAttention(input_dim= f_input_dim, num_heads=f_num_heads, output_dim=EMBED_DIM).to(device) 

def extract_3d_attention_vectors(batch):
    out, wei = pcd_attention_model(batch)
    return out
    
def extract_pedestrian_cloud_attention_vectors(batch):
    attention_vectors = pointnet_model(batch) 
    return attention_vectors

def extract_pedestrian_features_attention_vectors(batch):
    out, wei = featuers_attention_model(batch)
    return out


# Compute class weights
label_data   = pd.read_csv(LABEL_CSV_PATH)
class_counts = label_data['intended_actions'].value_counts()
num_neg, num_pos = class_counts[0], class_counts[1]

# Calculate weights
class_weights = torch.tensor([ num_neg / num_pos ], dtype=torch.float)

# print("class weights: ",class_weights)
model = AttentionFusionHead(vector_dim=EMBED_DIM, num_heads=NUM_HEADS).to(device)  # Initialize your model
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

parameters = chain(model.parameters(), pcd_attention_model.parameters(), pointnet_model.parameters())
optimizer = torch.optim.Adam(params=parameters, lr=0.001)

print('Number of parameters:',sum(p.numel() for p in model.parameters()), " parameters")
print('model device:', next(model.parameters()).device)

CHECKPOINT_DIR = SCRIPT_PATH / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"
LAST_CHECKPOINT_PATH = CHECKPOINT_DIR / "last_checkpoint.pth"

# Resume logic
START_EPOCH = 0
BEST_VAL_ACC = 0.0
if LAST_CHECKPOINT_PATH.exists():
    checkpoint = torch.load(LAST_CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    START_EPOCH = checkpoint['epoch']
    BEST_VAL_ACC = checkpoint.get('val_acc', 0.0)
    print(f"Resuming from epoch {START_EPOCH}, best val acc: {BEST_VAL_ACC:.2f}%")


def train_batch(model, criterion, optimizer):
    model.train()
    total_loss = 0.0

    correct = 0  # Track correct predictions
    total = 0
    
    progress_bar = tqdm(train_dl, desc="Training", leave=False)  # Initialize tqdm progress bar

    for batch in progress_bar:
        batched_avatar_points, batched_reconstructed_environment, batched_pedestrian_features, target = batch

        source1 = extract_3d_attention_vectors(batched_reconstructed_environment)  # (num_pedestrians, seq_len, vector_dim)
        source2 = extract_pedestrian_cloud_attention_vectors(batched_avatar_points)
        source3 = extract_pedestrian_features_attention_vectors(batched_pedestrian_features)

        # Forward pass
        outputs = model(source1, source2, source3)

        # Compute loss
        target = target.view(-1).float()
        loss = criterion(outputs.view(-1), target)

        # Compute accuracy for binary classification
        preds = (torch.sigmoid(outputs) > 0.5).int()
        target = target.int()
        correct += (preds == target).sum().item()
        total += target.size(0)


        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Update tqdm progress bar
        progress_bar.set_postfix(loss=loss.item(), acc=(correct / total))

    average_loss = total_loss / len(train_dl)
    accuracy =  correct / total if total > 0 else 0  # Compute final accuracy

    return average_loss, accuracy


@torch.no_grad()
def val_batch(model, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in val_dl:
        batched_avatar_points, batched_reconstructed_environment, batched_pedestrian_features, target = batch

        source1 = extract_3d_attention_vectors(batched_reconstructed_environment)  # (num_pedestrians, seq_len, vector_dim)
        source2 = extract_pedestrian_cloud_attention_vectors(batched_avatar_points)
        source3 = extract_pedestrian_features_attention_vectors(batched_pedestrian_features)

        # Forward pass  
        outputs = model(source1, source2, source3)

        # Compute loss
        target = target.view(-1).float()
        loss = criterion(outputs.view(-1), target)

        total_loss += loss.item()
        # Compute accuracy
        preds = (torch.sigmoid(outputs) > 0.5).int()
        target = target.int()
        correct += (preds == target).sum().item()
        total += target.size(0)


    average_loss = total_loss / len(val_dl)
    accuracy = correct / total  # Compute validation accuracy
    return average_loss,accuracy     
    


# Training loop with model saving
BEST_VAL_ACC = 0.0
# log = Report(N_EPOCHS)

for epoch in range(START_EPOCH, N_EPOCHS):
    print(f"\nEpoch {epoch+1}/{N_EPOCHS}")

    # Train
    train_loss, train_acc = train_batch(model, criterion, optimizer)

    # Validate
    val_loss, val_acc = val_batch(model, criterion)

    print(f"Epoch {epoch+1}/{N_EPOCHS} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # log.record(epoch+1, trn_loss=train_loss, trn_acc=train_acc, val_loss=val_loss, val_acc=val_acc)

    # Save model checkpoint every n epochs
    if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
        # checkpoint_filename = CHECKPOINT_PATH.format(epoch=epoch + 1)
        # torch.save(model.state_dict(), checkpoint_filename)
        # print(f"Checkpoint saved: {checkpoint_filename}")
        torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc
        }, LAST_CHECKPOINT_PATH)

    # Save best model
    if val_acc > BEST_VAL_ACC:
        BEST_VAL_ACC = val_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"New best model saved with val_acc: {val_acc:.2f}%")

# Save final model
# torch.save(model.state_dict(), FINAL_MODEL_PATH)
# print(f"Final model saved: {FINAL_MODEL_PATH}")

# Plot loss & accuracy
# log.plot_epochs(['trn_loss', 'val_loss', 'trn_acc', 'val_acc'])
