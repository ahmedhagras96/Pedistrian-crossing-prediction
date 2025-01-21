import ast
from dataclasses import dataclass

from modules.config.config_loader import ConfigLoader


@dataclass
class Reconstuction3DConfig:
    """
    Configuration parameters for point cloud processing.
    """
    frame_step: int = ConfigLoader.get("reconstruction.frame_step")
    frames_max_threshold: int = ConfigLoader.get("reconstruction.frames_max_threshold")
    tracked_objects: tuple[str] = ast.literal_eval(ConfigLoader.get("reconstruction.tracked_objects"))
    voxel_size: float = ConfigLoader.get("reconstruction.voxel_size")
    use_downsampling: bool = ConfigLoader.get("reconstruction.use_downsampling")
    point_size: float = ConfigLoader.get("reconstruction.point_size")
    background_color: tuple[float] = ast.literal_eval(ConfigLoader.get("reconstruction.background_color"))
