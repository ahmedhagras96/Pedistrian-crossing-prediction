from features_extraction.pedistrian_featuers_extraction import extract_pedistrian_featuers
# ##########Preprocessing############
# ##### Pedistrian Avatars #####
# # Define the paths directly
root_dir = 'D:\AUC_RA\pedestrian behavior prediction using pointcloud with images\Pedistrian-crossing-prediction\LOKI'

featuers_output_directory = 'D:\AUC_RA\pedestrian behavior prediction using pointcloud with images\Pedistrian-crossing-prediction\\training_data\pedistrian_featuers'
extract_pedistrian_featuers(root_dir,featuers_output_directory)