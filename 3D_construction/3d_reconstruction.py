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
        scenario_path=os.path.join(loki_path, 'scenario_026'),
        loki_csv_path=os.path.join(loki_path, 'loki.csv'),
    )

    # This is the format that is supported by the MapAligner
    ID_to_FR = {
        "4ff8af4d-6840-47c2-bc9b-eb383009ad65":[0,2,8],
        "624e3a59-7b6f-4674-a223-41966cdfa39a":[6,10],
        "6415887d-665b-40d7-8676-99074d06be6d":[64,66]
        }
    SavePath = os.path.join(loki_path, 'scenario_026_reconstructed')

    map_aligner.align(ID_to_FR=ID_to_FR, save=True, use_downsampling=True ,save_path=SavePath, scale = 20)
   


def main():
    odometry_aligner_test()
    pedestrian_map_aligner_test()


if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    Logger.configure_unified_file_logging(log_file)
    logger = Logger.get_logger(__name__)
    main()
