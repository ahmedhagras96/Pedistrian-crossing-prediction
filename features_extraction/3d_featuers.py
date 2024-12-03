import os
import math
import pandas as pd
import copy
from utils import *


def process_scenario(scenario_path, scenario_name):
    frame_files = sorted(os.listdir(scenario_path))
    odom_files = [f for f in frame_files if f.startswith("odom")]
    label_files = [f for f in frame_files if f.startswith("label3d")]
    
    results = []
    previous_positions = {}
    # to_remove_unknown = set()  # Track pedestrians with unknown movement to be removed later
    pending_corrections = {}
    perv_ego_postion = {}

    for odom_file, label_file in zip(odom_files, label_files):
        frame_id = int(odom_file.split("_")[1].split(".")[0])
        odom_path = os.path.join(scenario_path, odom_file)
        label_path = os.path.join(scenario_path, label_file)

        ego_position = parse_odometry(odom_path)
        pedestrians = parse_labels(label_path)

        current_results = []
        
        for ped_id, ped_data in pedestrians.items():
            ped_x, ped_y = ped_data["x"], ped_data["y"]
            
            # Ignore pedestrians behind the car
            if ped_x < 0:
                continue

            # Calculate distance from the ego car
            distance = math.sqrt(ped_x**2 + ped_y**2)
            
            # Determine movement
            if ped_id in previous_positions:
                prev_x, prev_y = previous_positions[ped_id]

                ped_modify_factor = math.sqrt((ego_position['x'] - perv_ego_postion['x'])**2 + (ego_position['y'] - perv_ego_postion['y'])**2)
                movement = math.sqrt((ped_x - prev_x)**2 + (ped_y - prev_y)**2)-ped_modify_factor

                # print(f"frame id: {frame_id} and ped_id is : {ped_id} movement: ",movement)
                movement_status = "Stopped" if movement < 0.25 else "Moving"

                if ped_id in pending_corrections:
                    results = [record for record in results if not (record["frame_id"] == pending_corrections[ped_id] and record["pedestrian_id"] == ped_id)]
                    del pending_corrections[ped_id]
            else:
                movement = None
                movement_status = "Unknown"
                # to_remove_unknown.add(ped_id)
                pending_corrections[ped_id] = frame_id

            # Save the current position for the next frame
            previous_positions[ped_id] = (ped_x, ped_y)
            
            # Append current results
            current_results.append({
                "scenario": scenario_name,
                "frame_id": frame_id,
                "pedestrian_id": ped_id,
                "distance": distance,
                "movement_status": movement_status,
            })

        perv_ego_postion = copy.copy(ego_position)
        # print("perv_ego_postion: ",perv_ego_postion)

        # Add current results to the overall list
        results.extend(current_results)

        # Remove pedestrians that are no longer in the current frame
        previous_positions = {pid: pos for pid, pos in previous_positions.items() if pid in pedestrians}
    return results

def process_all_scenarios(root_directory):
    results = []
    for scenario in os.listdir(root_directory):
        scenario_path = os.path.join(root_directory, scenario)
        if not os.path.isdir(scenario_path):
            continue

        # Process the scenario directory
        results.extend(process_scenario(scenario_path, scenario))

    # Write results to a single CSV
    df = pd.DataFrame(results)
    df.to_csv("pedestrian_analysis.csv", index=False)

# Run the processing function
root_directory = "../Loki_Dataset_sample"
process_all_scenarios(root_directory)

print("Analysis complete. Results saved to 'pedestrian_analysis.csv'.")
