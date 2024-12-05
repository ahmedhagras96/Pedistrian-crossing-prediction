# modules/config.py
from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class Config:
    """
    Configuration parameters for point cloud processing.
    """
    frame_step: int = 2
    frames_max_threshold: int = 100
    voxel_size: float = 0.05
    use_downsampling: bool = True
    point_size: float = 3.0
    background_color: List[float] = field(default_factory=lambda: [0.2, 0.2, 0.2])


@dataclass
class Arguments:
    """
    Command-line arguments for the processor.
    """
    scenario: str = 'scenario_000'
    output_directory: str = os.path.join(os.path.dirname(__file__), '../output_directory/')
    loki_path: str = os.path.join(os.path.dirname(__file__), '../LOKI/')
    max_frames: int = 10
    alignment_interval: int = 10
    save_files: bool = True
