import torch
import torch.nn as nn
from .linear_fusion_head.attention_Base_fusion import AttentionFusionHead
    

def extract_3d_attention_vectors(scenario):
    ###assign the model that extract attention vectors from 3d construction here 
    ##and return the attention vectors according to the batch sent
    pass

def extract_pedestrian_cloud_attention_vectors(senario):
    pass

def extract_pedestrian_features_attention_vectors(scenario):
    pass

def get_labels_for_scenario(senario):
    pass

model = AttentionFusionHead(vector_dim=64, num_heads=4)  # Initialize your model
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
scenarios_len = 126
for epoch in range(100):  # Number of epochs
    model.train()
    total_loss = 0.0
    for scenario in scenarios_len:
        # Extract data for this scenario
        #pedistrian filtration layers
        

        source1 = extract_3d_attention_vectors(scenario)  # (num_pedestrians, seq_len, vector_dim)
        source2 = extract_pedestrian_cloud_attention_vectors(scenario)
        source3 = extract_pedestrian_features_attention_vectors(scenario)
        labels = get_labels_for_scenario(scenario)  # (num_pedestrians, 1)

        # Forward pass
        outputs = model(source1, source2, source3)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/100], Loss: {total_loss:.4f}")