from ped_3D_avatar.pedistrian_pc_avatar_extraction import PedestrianProcessingPipeline
from features_extraction.pedistrian_featuers_extraction import extract_pedistrian_featuers

# ##########Preprocessing############
# ##### Pedistrian Avatars #####
# # Define the paths directly
root_dir = 'D:\AUC_RA\pedestrian behavior prediction using pointcloud with images\Pedistrian-crossing-prediction\LOKI'
csv_path = 'D:\AUC_RA\pedestrian behavior prediction using pointcloud with images\Pedistrian-crossing-prediction\LOKI\\b_loki.csv'
threshold_multiplier = 0.5
output_csv = '../LOKI/pedestrian_pointclouds.csv'
avatar_output_directory = 'D:\AUC_RA\pedestrian behavior prediction using pointcloud with images\Pedistrian-crossing-prediction\\training_data\pedistrian_avatars'
# Create the pipeline instance
pipeline = PedestrianProcessingPipeline(
    root_dir=root_dir,
    csv_path=csv_path,
    save_dir=avatar_output_directory,
    threshold_multiplier=threshold_multiplier
)

try:

    # Load scenario and frame IDs
    scenario_ids, frame_ids = pipeline.load_scenario_frame_ids()

    # Verify scenarios
    valid_scenario_ids = pipeline.verify_scenarios(scenario_ids)

    # Verify frames
    valid_frame_ids = pipeline.verify_frames(valid_scenario_ids, frame_ids)

    # Process all valid frames & save pcds
    df_pedestrians_filtered = pipeline.process_all_frames_and_crop_pedestrians(valid_scenario_ids, valid_frame_ids)
    df_pedestrians_filtered.to_csv("avatar_filtered_pedistrians.csv")



    ##### Pedistrian fatuers #####
    featuers_output_directory = 'D:\AUC_RA\pedestrian behavior prediction using pointcloud with images\Pedistrian-crossing-prediction\\training_data\pedistrian_featuers'
    
    #added avatar_output_directory to filter the featuers accoding to files existing only in avatar directory 
    extract_pedistrian_featuers(root_dir,featuers_output_directory,avatar_output_directory)



    ##### Pedistrian 3D construction ####

except KeyboardInterrupt:
  print("\nOperation cancelled by user. Exiting gracefully...")