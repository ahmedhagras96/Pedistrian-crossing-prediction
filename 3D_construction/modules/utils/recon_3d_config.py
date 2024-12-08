from dataclasses import dataclass, field
from typing import List


@dataclass
class Reconstuction3DConfig:
    """
    Configuration parameters for point cloud processing.
    """
    frame_step: int = 2
    frames_max_threshold: int = 100
    tracked_objects: tuple[str] = ('Car', 'Pedestrian')  # Now immutable
    voxel_size: float = 0.05
    use_downsampling: bool = True
    point_size: float = 3.0
    background_color: tuple[float] = (0.2, 0.2, 0.2)
