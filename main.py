import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from itertools import chain
from torch_snippets.torch_loader import Report
from linear_fusion_head.attention_Base_fusion import AttentionFusionHead
from attention_vector.point_cloud_attn_vector import PointCloudAttentionModel
from attention_vector.Pedestrian_PC_Attention.Ped_Att_Model import PointNetFeatureExtractor

from DataLoader import get_data_loaders

# Define the paths
SCRIPT_PATH = Path(__file__).resolve().parent
LOKI_PATH = SCRIPT_PATH / "LOKI"
LABEL_CSV_PATH = LOKI_PATH / "b_avatar_filtered_pedistrians.csv"
PEDESTRIAN_DIR = LOKI_PATH / 'training_data' / 'pedistrian_avatars'
ENVIRONMENT_DIR = LOKI_PATH / 'training_data' / '3d_constructed'
FEATURE_DIR = LOKI_PATH / 'training_data' / 'pedistrian_featuers'

# Global params
BATCH_SIZE = 32
EMBED_DIM = 128
N_EPOCHS = 1

# Params for model (CAV & LWSA)
KERNEL_SIZE = 3
NUM_HEADS = 4

# Create data loaders
train_dl, val_dl, test_dl = get_data_loaders(
    PEDESTRIAN_DIR, ENVIRONMENT_DIR, FEATURE_DIR, LABEL_CSV_PATH,
    batch_size=BATCH_SIZE, train_set_percentage=0.7, val_set_percentage=0.2
)

# init attention model (CAV & LWSA)
pcd_attention_model = PointCloudAttentionModel(embed_dim=EMBED_DIM, kernel_size=KERNEL_SIZE, num_heads=NUM_HEADS)

pointnet_model = PointNetFeatureExtractor(input_dim=3, output_dim=EMBED_DIM)


def extract_3d_attention_vectors(batch):
    out, wei = pcd_attention_model(batch)
    return out
    
def extract_pedestrian_cloud_attention_vectors(batch):
    attention_vectors = pointnet_model(batch) 
    return attention_vectors

def extract_pedestrian_features_attention_vectors(batch):
    pass


# Compute class weights
label_data = pd.read_csv(LABEL_CSV_PATH)
class_counts = label_data['intended_actions'].value_counts()
total_samples = class_counts.sum()

# Calculate weights
class_weights = torch.tensor(
    [total_samples / (len(class_counts) * count) for count in class_counts],
    dtype=torch.float
)


model = AttentionFusionHead(vector_dim=EMBED_DIM, num_heads=NUM_HEADS)  # Initialize your model
criterion = nn.BCELoss(weight=class_weights)

parameters = chain(model.parameters(), pcd_attention_model.parameters(), pointnet_model.parameters())
optimizer = torch.optim.Adam(params=parameters, lr=0.001)

print('Number of parameters:',sum(p.numel() for p in model.parameters()), " parameters")



def train_batch(model, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for batch in train_dl:
        batched_avatar_points, batched_reconstructed_environment, batched_pedestrian_features, target = batch

        source1 = extract_3d_attention_vectors(batched_reconstructed_environment)  # (num_pedestrians, seq_len, vector_dim)
        source2 = extract_pedestrian_cloud_attention_vectors(batched_avatar_points)
        source3 = extract_pedestrian_features_attention_vectors(batched_pedestrian_features)

        # Forward pass
        outputs = model(source1, source2, source3)

        # Compute loss
        loss = criterion(outputs, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    average_loss = total_loss / len(train_dl)

    return average_loss


@torch.no_grad()
def val_batch(model, criterion):
    model.eval()
    total_loss = 0.0
    for batch in val_dl:
        batched_avatar_points, batched_reconstructed_environment, batched_pedestrian_features, target = batch

        source1 = extract_3d_attention_vectors(batched_reconstructed_environment)  # (num_pedestrians, seq_len, vector_dim)
        source2 = extract_pedestrian_cloud_attention_vectors(batched_avatar_points)
        source3 = extract_pedestrian_features_attention_vectors(batched_pedestrian_features)

        # Forward pass  
        outputs = model(source1, source2, source3)

        # Compute loss
        loss = criterion(outputs, target)

        total_loss += loss.item()
    average_loss = total_loss / len(val_dl)

    return average_loss     
    

log = Report(N_EPOCHS)
for epoch in range(N_EPOCHS):
  N=len(train_dl)

  for ix, _ in enumerate(train_dl):
    avg_loss = train_batch(model, criterion, optimizer)
    log.record(epoch+(ix+1)/N, trn_loss= avg_loss, end='\r')

  val_loss=0
  N=len(val_dl)
  for ix, _ in enumerate(val_dl):

    loss= val_batch(model, criterion)
    val_loss+= loss
    log.record(epoch+(ix+1)/N, val_loss=loss, end='\r')

  log.report_avgs(epoch+1)
log.plot_epochs(['trn_loss', 'val_loss'])
