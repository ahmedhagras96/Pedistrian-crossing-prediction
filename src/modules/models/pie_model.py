import torch
import torch.nn as nn

from modules.attention.pedestrian_attention_pipeline import run_pedestrian_attention_pipeline
from modules.attention.point_cloud_attention_pipeline import run_point_cloud_attention_pipeline
from modules.utilities.logger import LoggerUtils


def train_pie_model():
    # Initialize logger
    logger = LoggerUtils.get_logger(__name__)

    # Define model
    model = None  # TODO: replace with attention fusion head
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    scenarios_len = 126
    for epoch in range(100):
        model.train()
        total_loss = 0.0

        for scenario in range(scenarios_len):
            # TODO: Extract data for this scenario
            # TODO: pedistrian filtration layers

            construction_attention = run_point_cloud_attention_pipeline()
            pedestrian_attention = run_pedestrian_attention_pipeline()
            features_attention = None  # TODO: add features attention

            labels = None  # TODO: get labels

            # Forward pass
            output = model(construction_attention, pedestrian_attention, features_attention)

            # Compute loss
            loss = criterion(output, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        logger.info(f"Epoch [{epoch + 1}/100], Loss: {total_loss:.4f}")
