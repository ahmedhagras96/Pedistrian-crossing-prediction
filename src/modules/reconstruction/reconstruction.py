import os
from pathlib import Path

from modules.config.logger import Logger
from modules.config.paths_loader import PATHS
from modules.reconstruction.aligners.align_direction import AlignDirection
from modules.reconstruction.aligners.odometry_aligner import PointCloudOdometryAligner
from modules.reconstruction.aligners.pedestrian_map_aligner import PedestrianMapAligner
from modules.reconstruction.utilities.visualization import PointCloudVisualizer

#: Directory path containing raw data
LOKI_PATH = PATHS.RAW_DATA_PATH

#: CSV file path for Loki data
LOKI_CSV_PATH = PATHS.LOKI_CSV_FILE

#: Module-level logger for this script
logger = Logger.get_logger("Reconstruction")


def run_odometry_aligner_pipeline() -> None:
    """
    Demonstrates how to use the PointCloudOdometryAligner to align environment and object point clouds
    via odometry data. Logs informational messages and visualizes the results.

    Raises:
        RuntimeError: Propagates from PointCloudOdometryAligner or PointCloudVisualizer 
            if alignment or visualization fails.
    """
    logger.info("Running Odometry Aligner Test...")

    odom_aligner = PointCloudOdometryAligner(
        scenario_path=os.path.join(LOKI_PATH, 'scenario_000'),
        loki_csv_path=LOKI_CSV_PATH,
    )

    # Align the point clouds; adjust arguments as appropriate for your scenario
    odometry_environment, odometry_objects = odom_aligner.align(
        key_frame=20, align_interval=10, align_direction=AlignDirection.SPLIT
    )

    logger.info(
        f"Odometry Aligned Environment Point Cloud has {len(odometry_environment.points)} points."
    )
    logger.info(
        f"Odometry Aligned Objects Point Cloud has {len(odometry_objects.points)} points."
    )

    # Visualize the aligned points
    visualizer = PointCloudVisualizer()
    visualizer.add_point_cloud(odometry_environment, [0.5, 0.5, 0.5])  # Gray
    visualizer.add_point_cloud(odometry_objects, color=[1.0, 0.0, 0.0])  # Red
    visualizer.run()
    visualizer.close()


def run_pedestrian_map_aligner_pipeline() -> None:
    """
    Demonstrates how to use the PedestrianMapAligner to align pedestrian point clouds 
    and optionally save them to disk.

    Raises:
        RuntimeError: Propagates from PedestrianMapAligner if alignment or file saving fails.
    """
    logger.info("Running Pedestrian Map Aligner Test...")

    map_aligner = PedestrianMapAligner(
        loki_csv_path=PATHS.AVATAR_FILTERED_PEDESTRIANS_CSV_FILE,
        data_path=LOKI_PATH
    )

    save_path = PATHS.RECONSTRUCTED_DATA_PATH
    # Adjust arguments to align/save as needed for your scenario
    map_aligner.align(
        save=True,
        use_downsampling=True,
        save_path=save_path,
        scaling_factor=20
    )


def main() -> None:
    """
    Runs all reconstruction tests in sequence.
    
    Raises:
        RuntimeError: If either test function fails internally.
    """
    run_odometry_aligner_pipeline()
    run_pedestrian_map_aligner_pipeline()


if __name__ == '__main__':
    # Configure unified logging to write to a file
    Logger.configure_unified_logging_file(PATHS.LOGS_PATH / Path("reconstruction.log"))

    # Run the main entry point
    main()
