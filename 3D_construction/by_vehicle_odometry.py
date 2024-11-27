import os
import argparse
import open3d as o3d
import numpy as np

# Configuration Parameters
CONFIG = {
    'frame_step': 2,                # Step size for frames, e.g., 2 for even frames only
    'voxel_size': 0.05,             # Voxel downsample size
    'use_downsampling': True,       # Toggle downsampling on/off
    'point_size': 3.0,              # Point size in visualization
    'background_color': [1, 1, 1],  # Background color for visualization (white)
}

def load_vehicle_odometry(file_path):
    """Load vehicle odometry data from a file."""
    with open(file_path, 'r') as file:
        data = file.readline().strip().split(',')
        if len(data) < 6:
            raise ValueError(f"Odometry file {file_path} is malformed.")
          
        x, y, z = map(float, data[:3])
        roll, pitch, yaw = map(float, data[3:])
        
    return x, y, z, roll, pitch, yaw

def apply_vehicle_transformation(pc, vehicle_odometry):
    """Apply transformation to point cloud using vehicle odometry."""
    x, y, z, roll, pitch, yaw = vehicle_odometry
    translation = np.array([x, y, z])
    rotation = o3d.geometry.get_rotation_matrix_from_xyz((roll, pitch, yaw))
    
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    pc.transform(transformation_matrix)
    
    return pc

def parse_labels(label_file):
    """Parse label file to extract object data."""
    objects = []
    with open(label_file, 'r') as file:
        next(file)  # Skip header
        for line in file:
            parts = line.strip().split(',')
            
            if len(parts) < 10:
                continue
              
            try:
                label, track_id = parts[0], parts[1]
                pos = tuple(map(float, parts[3:6]))
                dims = tuple(map(float, parts[6:9]))
                yaw = float(parts[9])
                objects.append({'label': label, 'track_id': track_id, 'position': pos, 'dimensions': dims, 'yaw': yaw})
            except ValueError:
                continue
              
    return objects

def crop_objects_from_pcd(pc, objects, include_labels=None):
    """Crop specific objects from point cloud."""
    cropped_pcs = []
    for obj in objects:
        if include_labels and obj['label'].lower() not in include_labels:
            continue
        pos, dims, yaw = obj['position'], obj['dimensions'], obj['yaw']
        
        box = o3d.geometry.OrientedBoundingBox(
            center=np.array(pos),
            R=o3d.geometry.get_rotation_matrix_from_xyz([0, 0, yaw]),
            extent=np.array(dims)
        )
        cropped = pc.crop(box)
        cropped_pcs.append(cropped)
        
    if cropped_pcs:
        combined = cropped_pcs[0]
        for cpc in cropped_pcs[1:]:
            combined += cpc
        return combined
      
    return None

def crop_environment(pc, objects):
    """Remove all labeled objects, leaving only the environment."""
    for obj in objects:
        pos, dims, yaw = obj['position'], obj['dimensions'], obj['yaw']
        box = o3d.geometry.OrientedBoundingBox(
            center=np.array(pos),
            R=o3d.geometry.get_rotation_matrix_from_xyz([0, 0, yaw]),
            extent=np.array(dims)
        )
        pc = pc.crop(box, invert=True)
        
    return pc

def save_point_cloud(pc, save_path):
    """Save point cloud to file."""
    try:
        o3d.io.write_point_cloud(save_path, pc)
        print(f"Point cloud saved to: {save_path}")
    except Exception as e:
        print(f"Failed to save point cloud to {save_path}: {e}")

def visualize_point_clouds(environment_pc, cropped_pc, point_size, background_color):
    """Visualize environment and cropped objects."""
    if not environment_pc and not cropped_pc:
        print("No valid point clouds found to display.")
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    if environment_pc:
        environment_pc.paint_uniform_color([0.5, 0.5, 0.5])  # Gray
        vis.add_geometry(environment_pc)

    if cropped_pc:
        cropped_pc.paint_uniform_color([1, 0, 0])  # Red
        vis.add_geometry(cropped_pc)

    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.background_color = np.asarray(background_color)

    vis.run()
    vis.destroy_window()

def process_scenario(args):
    """Process the scenario to align the environment and cropped objects."""
    scenario_path = os.path.join(args.loki_path, args.scenario)
    save_directory = args.output_directory
    max_frames, frame_step = args.max_frames, CONFIG['frame_step']
    voxel_size, use_downsampling = CONFIG['voxel_size'], CONFIG['use_downsampling']
    alignment_interval = args.alignment_interval
    save_files = args.save_files

    if save_files:
        os.makedirs(save_directory, exist_ok=True)

    aligned_cropped_pcs = []
    environment_queue = []
    last_positions = {}

    # Pass 1: Identify the last frame where each object appears
    frame_index = 0
    processed_frames = 0
    while processed_frames < max_frames:
        label_file = os.path.join(scenario_path, f'label3d_{frame_index:04d}.txt')

        if os.path.isfile(label_file):
            try:
                objects = parse_labels(label_file)
                for obj in objects:
                    track_id = obj['track_id']
                    last_positions[track_id] = {'frame': frame_index, 'object': obj}
                processed_frames += 1
            except Exception as e:
                print(f"Error parsing labels from {label_file}: {e}")

        frame_index += frame_step

    # Pass 2: Process the frames for alignment
    frame_index = 0
    processed_frames = 0
    combined_environment_pc = None

    while processed_frames < max_frames:
        pc_file = os.path.join(scenario_path, f'pc_{frame_index:04d}.ply')
        odom_file = os.path.join(scenario_path, f'odom_{frame_index:04d}.txt')
        label_file = os.path.join(scenario_path, f'label3d_{frame_index:04d}.txt')

        if not (os.path.isfile(pc_file) and os.path.isfile(odom_file) and os.path.isfile(label_file)):
          continue
            
        try:
            pc = o3d.io.read_point_cloud(pc_file)
            if use_downsampling:
              pc = pc.voxel_down_sample(voxel_size=voxel_size)
        except Exception as e:
            print(f"Failed to read point cloud from {pc_file}: {e}")
            frame_index += frame_step
            continue

        try:
            objects = parse_labels(label_file)
        except Exception as e:
            print(f"Error parsing labels from {label_file}: {e}")
            frame_index += frame_step
            continue

        # Select objects appearing in their last frame
        cropped_objects = [last_positions[track_id]['object'] for track_id in last_positions if last_positions[track_id]['frame'] == frame_index]
        environment_pc = crop_environment(pc, objects)
        cropped_pc = crop_objects_from_pcd(pc, cropped_objects)

        try:
            vehicle_odometry = load_vehicle_odometry(odom_file)
        except Exception as e:
            print(f"Failed to load odometry from {odom_file}: {e}")
            frame_index += frame_step
            continue

        if environment_pc:
            try:
                aligned_env_pc = apply_vehicle_transformation(environment_pc, vehicle_odometry)
                environment_queue.append(aligned_env_pc)
                if len(environment_queue) > alignment_interval:
                    environment_queue.pop(0)

                combined_environment_pc = environment_queue[0]
                for env_pc in environment_queue[1:]:
                    combined_environment_pc += env_pc
            except Exception as e:
                print(f"Failed to apply transformation to environment point cloud: {e}")

        if cropped_pc:
            try:
                aligned_cropped_pc = apply_vehicle_transformation(cropped_pc, vehicle_odometry)
                aligned_cropped_pcs.append(aligned_cropped_pc)
            except Exception as e:
                print(f"Failed to apply transformation to cropped objects point cloud: {e}")

        processed_frames += 1

        frame_index += frame_step

    # Combine all aligned cropped objects into a single point cloud
    if aligned_cropped_pcs:
        combined_cropped_pc = aligned_cropped_pcs[0]
        for pc in aligned_cropped_pcs[1:]:
            combined_cropped_pc += pc
    else:
        combined_cropped_pc = None

    # Merge environment and cropped objects
    if combined_environment_pc and combined_cropped_pc:
        combined_pcd = combined_environment_pc + combined_cropped_pc
    elif combined_environment_pc:
        combined_pcd = combined_environment_pc
    elif combined_cropped_pc:
        combined_pcd = combined_cropped_pc
    else:
        combined_pcd = None

    # Save combined point clouds if saving is enabled
    if save_files and combined_pcd:
        combined_save_path = os.path.join(save_directory, "combined_aligned.pcd")
        save_point_cloud(combined_pcd, combined_save_path)

    # Visualize the combined point clouds
    visualize_point_clouds(combined_environment_pc, combined_cropped_pc, CONFIG['point_size'], CONFIG['background_color'])

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process and visualize vehicle point clouds.")
    parser.add_argument('--scenario', type=str, default='scenario_000', help='Scenario identifier')
    
    parser.add_argument('--output_directory', type=str, default=os.path.join(os.path.dirname(__file__), './output_directory/'),
                        help='Directory to save processed point cloud files (default: ./output_directory/)')
    parser.add_argument('--loki_path', type=str, default=os.path.join(os.path.dirname(__file__), '../LOKI/'),
                        help='Base path for the LOKI data (default: ../LOKI/)')
    
    parser.add_argument('--max_frames', type=int, default=10, help='Maximum number of frames to process')
    parser.add_argument('--alignment_interval', type=int, default=10, help='Frames to keep for alignment')
    parser.add_argument('--save_files', action='store_true', help='Enable saving point clouds')
    return parser.parse_args()

def main():
    args = parse_arguments()
    process_scenario(args)

if __name__ == "__main__":
    main()
