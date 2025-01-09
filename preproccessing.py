import sys
import os

from ped_3D_avatar.pedistrian_pc_avatar_extraction import PedestrianProcessingPipeline
from features_extraction.pedistrian_featuers_extraction import extract_pedistrian_featuers

from reconstruction.modules.pedestrian_map_aligner import PedestrianMapAligner
from reconstruction.modules.utils.logger import Logger

import time
# ##########Preprocessing############
# ##### Pedistrian Avatars #####
# # Define the paths directly
script_path = os.path.normpath(os.path.abspath(os.path.dirname(__file__)))
loki_path = os.path.normpath(os.path.abspath(os.path.join(script_path, "LOKI")))
# print(loki_path)
log_file = os.path.normpath(os.path.abspath(os.path.join(script_path, "logs", "logs.log")))
# root_dir = 'D:\AUC_RA\pedestrian behavior prediction using pointcloud with images\Pedistrian-crossing-prediction\LOKI'
# csv_path = 'D:\AUC_RA\pedestrian behavior prediction using pointcloud with images\Pedistrian-crossing-prediction\LOKI\\b_loki.csv'
threshold_multiplier = 0.5
output_csv = os.path.join(loki_path, 'pedestrian_pointclouds.csv') #'../LOKI/pedestrian_pointclouds.csv'
avatar_output_directory = os.path.join(loki_path, 'training_data', 'pedistrian_avatars')

# Create the pipeline instance
pipeline = PedestrianProcessingPipeline(
    root_dir=loki_path,
    csv_path=os.path.join(loki_path, 'loki.csv'),
    save_dir=avatar_output_directory,
    threshold_multiplier=threshold_multiplier
)

# creating pedestrian map alignment object to perfomr 3D construction using the PedestrianMapAligner class.

# intialize logger
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
Logger.configure_unified_file_logging(log_file)
logger = Logger.get_logger(__name__)
def pedestrian_map_aligner():
    logger.info('Running Pedestrian Map Aligner ...')
    
    map_aligner = PedestrianMapAligner(
        scenario_path=None, #os.path.join(loki_path, 'scenario_026'),
        loki_csv_path=os.path.join(loki_path, 'avatar_filtered_pedistrians.csv'),
        data_path=loki_path
    )

    SavePath = os.path.join(loki_path, 'training_data', '3d_constructed')

    map_aligner.align(save=True, use_downsampling=True, save_path=SavePath, scaling_factor = 20)

try:

    # Load scenario and frame IDs
    scenario_ids, frame_ids = pipeline.load_scenario_frame_ids()

    # Verify scenarios
    valid_scenario_ids = pipeline.verify_scenarios(scenario_ids)

    # Verify frames
    valid_frame_ids = pipeline.verify_frames(valid_scenario_ids, frame_ids)

    # Process all valid frames & save pcds
    sart_time = time.time()
    df_pedestrians_filtered = pipeline.process_all_frames_and_crop_pedestrians(valid_scenario_ids, valid_frame_ids)
    df_pedestrians_filtered.to_csv(os.path.join(loki_path, 'avatar_filtered_pedistrians.csv'))
    end_time = time.time()
    print(f"Time taken to process all frames and crop pedestrians: {end_time - sart_time} seconds")


    #### Pedistrian fatuers #####
    featuers_output_directory =  os.path.join(loki_path, 'training_data', 'pedistrian_featuers')#'D:\AUC_RA\pedestrian behavior prediction using pointcloud with images\Pedistrian-crossing-prediction\\training_data\pedistrian_featuers'
    
    # added avatar_output_directory to filter the featuers accoding to files existing only in avatar directory 
    extract_pedistrian_featuers(loki_path,featuers_output_directory,avatar_output_directory)



    ##### Pedistrian 3D construction ####

    pedestrian_map_aligner()

except KeyboardInterrupt:
  print("\nOperation cancelled by user. Exiting gracefully...")