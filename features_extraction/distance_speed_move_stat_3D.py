import os
import math
import json
import pandas as pd
import copy
from .utils import *

def get_speed_dist_ms_scenario(scenario_path,fps):
    """
    Process a single scenario to extract features for each pedestrian,
    including corrections for initial "Unknown" movement status.
    """
    frame_files = sorted(os.listdir(scenario_path))
    odom_files = [f for f in frame_files if f.startswith("odom")]
    label_files = [f for f in frame_files if f.startswith("label3d")]

    # Track data
    previous_positions = {}
    pending_corrections = {}
    perv_ego_position = None

    scenario_features = {}

    for odom_file, label_file in zip(odom_files, label_files):
        cleaned_name = odom_file.split("_")[1].split(".")[0].split()[0]
        frame_id = int(cleaned_name)

        odom_path = os.path.join(scenario_path, odom_file)
        label_path = os.path.join(scenario_path, label_file)

        # Parse current odometry and labels
        ego_position = parse_odometry(odom_path)
        pedestrians = parse_labels(label_path)

        # Collect features for this frame
        frame_features = {}

        for ped_id, ped_data in pedestrians.items():
            ped_x, ped_y = ped_data["x"], ped_data["y"]

            # Ignore pedestrians behind the ego vehicle
            if ped_x < 0:
                continue

            if ped_id in previous_positions:
                # Calculate speed, distance, and movement status
                prev_position = previous_positions[ped_id]
                speed, distance, movement_status = calculate_speed_distance_movement(
                    prev_position, (ped_x, ped_y),perv_ego_position,ego_position,fps
                )

                # Remove pending corrections for this pedestrian if applicable
                if ped_id in pending_corrections:
                    pending_frame_id = pending_corrections[ped_id]
                    del scenario_features[pending_frame_id][ped_id]
                    del pending_corrections[ped_id]

            else:
                # Initialize for the first frame
                speed, distance = 0, math.sqrt(ped_x**2 + ped_y**2)
                movement_status = -1  #Unknown

                # Add this pedestrian to pending corrections
                pending_corrections[ped_id] = frame_id

            # Save current position for next frame
            previous_positions[ped_id] = (ped_x, ped_y)

            # Store features for this pedestrian
            frame_features[ped_id] = {
                "speed": speed,
                "distance": distance,
                "movement_status": movement_status
            }

        # Update scenario features
        scenario_features[frame_id] = frame_features

        # Update previous ego position
        perv_ego_position = ego_position

    return scenario_features


# def process_all_scenarios(root_directory,output_directory):
#     """
#     Process all scenarios in the dataset to get speed, distance and movment status for each
#     pedistrian and save results in JSON files in the `output_features` folder.
#     """
#     # Ensure the output directory exists
#     # output_directory = os.path.join(root_directory, "output_features")
#     os.makedirs(output_directory, exist_ok=True)

#     for scenario in os.listdir(root_directory):
#         scenario_path = os.path.join(root_directory, scenario)
#         if not os.path.isdir(scenario_path):
#             continue

#         print(f"Processing scenario: {scenario}...")
#         # Process the scenario
#         scenario_features = get_speed_dist_ms_scenario(scenario_path, scenario,5)

#         # Save scenario features to a JSON file
#         output_path = os.path.join(output_directory, f"{scenario}_features.json")
#         with open(output_path, "w") as json_file:
#             json.dump(scenario_features, json_file, indent=4)

#         print(f"Saved features for {scenario} to {output_path}")



# # Run the processing function
# output_directory = "./processed_scenarios/output_features_Speed & Distance"
# root_directory = "LOKI"
# process_all_scenarios(root_directory,output_directory)
