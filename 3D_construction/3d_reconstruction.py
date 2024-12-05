import sys
import os

from modules.odometry_aligner import PointCloudOdometryAligner, AlignDirection
from modules.utils.visualization import PointCloudVisualizer

def main():
    oaligner = PointCloudOdometryAligner(
        scenario_path=r'S:\researchlab\auc-research\Pedistrian-crossing-prediction\LOKI\scenario_000',
        loki_csv_path=r'S:\researchlab\auc-research\Pedistrian-crossing-prediction\LOKI\loki.csv',
        key_frame=30
    ) 
    
    odometry_environment, odometry_objects = oaligner.align(20, 10, AlignDirection.SPLIT)
    
    vis = PointCloudVisualizer()
    vis.add_point_cloud(odometry_environment, [0.5, 0.5, 0.5])
    vis.add_point_cloud(odometry_objects, color=[1, 0, 0])
    vis.run()
    vis.close()

if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    main()