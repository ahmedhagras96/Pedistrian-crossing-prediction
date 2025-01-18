from modules.config.config_loader import ConfigLoader
from modules.config.logger import LoggerUtils

from modules.config.paths_loader import PATHS
from modules.model.loaders.data_loader import DataLoaderUtils
from modules.model.training.trainer import Trainer

def run_training():
    # PATHS
    AVATARS_DATA_DIR=PATHS.AVATARS_DATA
    RECONSTRUCTION_DATA_DIR=PATHS.RECONSTRUCTION_DATA
    PEDESTRIANS_FEATURES_DATA_DIR=PATHS.PEDESTRIAN_FEATURES_DATA
    BIN_FILTERED_PEDESTRIANS_CSV=PATHS.BIN_FILTERED_PEDESTRIANS_CSV_PATH
    
    # Hyperparameters
    BATCH_SIZE = ConfigLoader.get("model.batch_size")
    TRAIN_SIZE = ConfigLoader.get("model.train_set_ratio")
    VAL_SIZE = ConfigLoader.get("model.validation_set_ratio")
    EMBED_DIM = ConfigLoader.get("model.embed_dim")
    KERNEL_SIZE = ConfigLoader.get("model.kernel_size")
    NUM_HEADS = ConfigLoader.get("model.num_heads")
    NUM_EPOCHS = ConfigLoader.get("model.num_epochs")
    
    # Initialize logger
    logger = LoggerUtils.get_logger("Main")
    
    # Create data loaders
    # Create data loaders
    train_dl, val_dl, test_dl = DataLoaderUtils.get_data_loaders(
        pedestrian_dir=AVATARS_DATA_DIR,
        environment_dir=RECONSTRUCTION_DATA_DIR,
        feature_dir=PEDESTRIANS_FEATURES_DATA_DIR,
        label_csv_path=BIN_FILTERED_PEDESTRIANS_CSV,
        batch_size=BATCH_SIZE, train_set_percentage=TRAIN_SIZE, val_set_percentage=VAL_SIZE
    )
    
    # Initialize Trainer and run training
    trainer = Trainer(
        train_dl=train_dl,
        val_dl=val_dl,
        embed_dim=EMBED_DIM,
        kernel_size=KERNEL_SIZE,
        num_heads=NUM_HEADS,
        num_epochs=NUM_EPOCHS,
        logger=logger
    )
    
    trainer.run_training()
