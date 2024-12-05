import sys
import os

from modules.odometry_aligner import PointCloudOdometryAligner, AlignDirection

def main():
    oaligner = PointCloudOdometryAligner(
        scenario_path=r'S:\researchlab\auc-research\Pedistrian-crossing-prediction\LOKI',
        loki_csv_path=r'S:\researchlab\auc-research\Pedistrian-crossing-prediction\LOKI\loki.csv',
        key_frame=30
    ) 
    
    oaligner.align(30, 10, AlignDirection.SPLIT)

if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    main()