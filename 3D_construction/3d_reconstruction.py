import sys
import os

from modules.odometry_aligner import PointCloudOdometryAligner
from modules.pedestrian_map_aligner import PedestrianMapAligner
from modules.helpers.align_direction import AlignDirection
from modules.utils.visualization import PointCloudVisualizer
from modules.utils.logger import Logger

script_path = os.path.normpath(os.path.abspath(os.path.dirname(__file__)))
loki_path = os.path.normpath(os.path.abspath(os.path.join(script_path, "..", "LOKI")))
log_file = os.path.normpath(os.path.abspath(os.path.join(script_path, "logs", "logs.log")))
logger = None

def odometry_aligner_test():
    logger.info('Running Odometry Aligner Test...')
    
    odom_aligner = PointCloudOdometryAligner(
        scenario_path=os.path.join(loki_path, 'scenario_000'),
        loki_csv_path=os.path.join(loki_path, 'loki.csv'),
    )

    odometry_environment, odometry_objects = odom_aligner.align(20, 10, AlignDirection.SPLIT)
    logger.info(f'Odometry Aligned Environment Point Cloud with {len(odometry_environment.points)} enviornment points')
    logger.info(f'Odometry Aligned Objects Point Cloud with {len(odometry_objects.points)} objects points')

    # Visualize the aligned points
    vis = PointCloudVisualizer()
    vis.add_point_cloud(odometry_environment, [0.5, 0.5, 0.5])
    vis.add_point_cloud(odometry_objects, color=[1, 0, 0])
    vis.run()
    vis.close()


def pedestrian_map_aligner_test():
    logger.info('Running Pedestrian Map Aligner Test...')
    
    map_aligner = PedestrianMapAligner(
        scenario_path=os.path.join(loki_path, 'scenario_000'),
        loki_csv_path=os.path.join(loki_path, 'loki.csv'),
    )

    map_environment, pedestrian, cars, scaled_box = map_aligner.align(frame_sequence=[38,40,42], pedestrian_id='4ab64275-275c-4f58-8ed5-39837a4a265d')
    logger.info(f'Map Aligned Environment Point Cloud with {len(map_environment.points)} enviornment points')
    logger.info(f'Map Aligned Objects Point Cloud with {len(pedestrian.points)} pedestrian points, {len(cars.points)} car points for the last frame of the sequence (for visualization)')
    
    # Visualize the aligned points
    visualizer = PointCloudVisualizer()
    visualizer.add_point_cloud(pedestrian, [0.5, 0.5, 1])
    visualizer.add_point_cloud(scaled_box)
    visualizer.add_point_cloud(cars, [1, 0.5, 0.5])
    visualizer.add_point_cloud(map_environment)
    visualizer.run()
    visualizer.close()

    # save 
    map_aligner.save(os.path.join(loki_path, 'scenario_000_reconstructed'), remove=False)


def main():
    # odometry_aligner_test()
    pedestrian_map_aligner_test()


if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    Logger.configure_unified_file_logging(log_file)
    logger = Logger.get_logger(__name__)
    main()
