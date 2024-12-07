# modules/pedestrian_map_aligner.py
import sys
import os

from modules.odometry_aligner import PointCloudOdometryAligner
from modules.pedestrian_map_aligner import PedestrianMapAligner
from modules.helpers.align_direction import AlignDirection
from modules.utils.visualization import PointCloudVisualizer

file_path = os.path.normpath(os.path.abspath(os.path.dirname(__file__)))
loki_path = os.path.normpath(os.path.abspath(os.path.join(file_path, "..", "LOKI")))

def odometry_aligner_test():
    oaligner = PointCloudOdometryAligner(
        scenario_path=os.path.join(loki_path, 'scenario_000'),
        loki_csv_path=os.path.join(loki_path, 'loki.csv'),
        key_frame=30
    )

    odometry_environment, odometry_objects = oaligner.align(20, 10, AlignDirection.SPLIT)

    vis = PointCloudVisualizer()
    vis.add_point_cloud(odometry_environment, [0.5, 0.5, 0.5])
    vis.add_point_cloud(odometry_objects, color=[1, 0, 0])
    vis.run()
    vis.close()
    
    
def pedestrian_map_aligner_test():
    maligner = PedestrianMapAligner(
        scenario_path=os.path.join(loki_path, 'scenario_000'),
        loki_csv_path=os.path.join(loki_path, 'loki.csv'),
        num_frames=1,
        pedestrian_id='4ab64275-275c-4f58-8ed5-39837a4a265d'
    )
    
    map_environment, pedestrian, cars, scaled_box = maligner.align()
    # Visualize the aligned points

    visualizer = PointCloudVisualizer()
    visualizer.add_point_cloud(pedestrian, [0.5, 0.5, 1])
    visualizer.add_point_cloud(scaled_box)
    visualizer.add_point_cloud(cars, [1, 0.5, 0.5])
    visualizer.add_point_cloud(map_environment)
    visualizer.run()
    visualizer.close()

    # save 
    # maligner.save(os.path.join(loki_path, 'scenario_026'), remove=False)


def main():
    # odometry_aligner_test()
    pedestrian_map_aligner_test()


if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    main()
