import os
import argparse
import open3d as o3d
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Optional
# from tqdm import tqdm


#! Change values in this class to reflect real changes
@dataclass
class Config:
    frame_step: int = 2
    frames_max_threshold: int = 100
    voxel_size: float = 0.05
    use_downsampling: bool = True
    point_size: float = 3.0
    background_color: List[float] = field(default_factory=lambda: [0.2, 0.2, 0.2])


#! Do not update values in this class, they will not reflect real changes
@dataclass
class Arguments:
    scenario: str = 'scenario_000'
    output_directory: str = os.path.join(os.path.dirname(__file__), './output_directory/')
    loki_path: str = os.path.join(os.path.dirname(__file__), '../LOKI/')
    max_frames: int = 10
    alignment_interval: int = 10
    save_files: bool = True


#! Change values in this class to reflect real changes
def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description="Process and visualize vehicle point clouds.")
    parser.add_argument('--scenario', type=str, default='scenario_000', help='Scenario identifier')
    parser.add_argument('--output_directory', type=str, default=os.path.join(os.path.dirname(__file__), './output_directory/'),
                        help='Directory to save processed point cloud files (default: ./output_directory/)')
    parser.add_argument('--loki_path', type=str, default=os.path.join(os.path.dirname(__file__), '../../LOKI/'),
                        help='Base path for the LOKI data (default: ../LOKI/)')
    parser.add_argument('--max_frames', type=int, default=30, help='Maximum number of frames to process')
    parser.add_argument('--alignment_interval', type=int, default=4, help='Frames to keep for alignment') # TODO: Fix even only inputs
    parser.add_argument('--save_files', default=True, action='store_true', help='Enable saving point clouds')
    args = parser.parse_args()

    return Arguments(
        scenario=args.scenario,
        output_directory=args.output_directory,
        loki_path=args.loki_path,
        max_frames=args.max_frames,
        alignment_interval=args.alignment_interval,
        save_files=args.save_files
    )

class PointCloudOdometryAligner:
    def __init__(self, args: Arguments, config: Config):
        # Initialize logging with detailed format
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)

        # Assign args and config
        self.args = args
        self.config = config
        self.scenario_path = os.path.normpath(os.path.join(args.loki_path, args.scenario))
        self.save_directory = os.path.normpath(args.output_directory)
        self.max_frames = args.max_frames
        self.frame_step = self.config.frame_step
        self.frames_max_threshold = self.config.frames_max_threshold
        self.voxel_size = self.config.voxel_size
        self.use_downsampling = self.config.use_downsampling
        self.alignment_interval = args.alignment_interval
        self.save_files = args.save_files
        self.point_size = self.config.point_size
        self.background_color = self.config.background_color

        self.logger.info("Initialized PointCloudOdometryAligner")
        self.logger.info(f"Scenario Path: {self.scenario_path}")
        self.logger.info(f"Output Directory: {self.save_directory}")
        self.logger.info(f"Max Frames: {self.max_frames}")
        self.logger.info(f"Frame Step: {self.frame_step}")
        self.logger.info(f"Voxel Size: {self.voxel_size}")
        self.logger.info(f"Use Downsampling: {self.use_downsampling}")
        self.logger.info(f"Alignment Interval: {self.alignment_interval}")
        self.logger.info(f"Save Files: {self.save_files}")

        if self.save_files:
            os.makedirs(self.save_directory, exist_ok=True)
            self.logger.info(f"Created output directory at: {self.save_directory}")

    def load_vehicle_odometry(self, file_path: str) -> tuple:
        """Load vehicle odometry data from a file."""
        self.logger.debug(f"Loading vehicle odometry from: {file_path}")
        with open(file_path, 'r') as file:
            data = file.readline().strip().split(',')
            if len(data) < 6:
                self.logger.error(f"Odometry file {file_path} is malformed.")
                raise ValueError(f"Odometry file {file_path} is malformed.")

            x, y, z = map(float, data[:3])
            roll, pitch, yaw = map(float, data[3:])

        self.logger.debug(f"Loaded odometry: x={x}, y={y}, z={z}, roll={roll}, pitch={pitch}, yaw={yaw}")
        return x, y, z, roll, pitch, yaw

    def apply_vehicle_transformation(self, pc: o3d.geometry.PointCloud, vehicle_odometry: tuple) -> o3d.geometry.PointCloud:
        """Apply transformation to point cloud using vehicle odometry."""
        self.logger.debug("Applying vehicle transformation to point cloud")
        x, y, z, roll, pitch, yaw = vehicle_odometry
        translation = np.array([x, y, z])
        rotation = o3d.geometry.get_rotation_matrix_from_xyz((roll, pitch, yaw))

        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation
        transformation_matrix[:3, 3] = translation
        pc.transform(transformation_matrix)

        self.logger.debug("Transformation applied successfully")
        return pc

    def parse_labels(self, label_file: str) -> List[dict]:
        """Parse label file to extract object data."""
        self.logger.debug(f"Parsing label file: {label_file}")
        objects = []
        try:
            with open(label_file, 'r') as file:
                next(file)  # Skip header
                for line in file:
                    parts = line.strip().split(',')

                    if len(parts) < 10:
                        self.logger.warning(f"Skipping malformed line in {label_file}: {line.strip()}")
                        continue

                    try:
                        label, track_id = parts[0], parts[1]
                        pos = tuple(map(float, parts[3:6]))
                        dims = tuple(map(float, parts[6:9]))
                        yaw = float(parts[9])
                        objects.append({
                            'label': label,
                            'track_id': track_id,
                            'position': pos,
                            'dimensions': dims,
                            'yaw': yaw
                        })
                    except ValueError as ve:
                        self.logger.warning(f"Value error parsing line in {label_file}: {line.strip()} - {ve}")
                        continue
        except FileNotFoundError:
            self.logger.error(f"Label file not found: {label_file}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error parsing label file {label_file}: {e}")
            raise

        self.logger.debug(f"Parsed {len(objects)} objects from {label_file}")
        return objects

    def crop_objects_from_pcd(self, pc: o3d.geometry.PointCloud, objects: List[dict], include_labels: Optional[List[str]] = None) -> Optional[o3d.geometry.PointCloud]:
        """Crop specific objects from point cloud."""
        self.logger.debug("Cropping objects from point cloud")
        cropped_pcs = []
        for obj in objects:
            if include_labels and obj['label'].lower() not in include_labels:
                self.logger.debug(f"Skipping object with label: {obj['label']}")
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
            self.logger.debug("Combined cropped point clouds")
            return combined

        self.logger.debug("No objects were cropped from the point cloud")
        return None

    def crop_environment(self, pc: o3d.geometry.PointCloud, objects: List[dict]) -> o3d.geometry.PointCloud:
        """Remove all labeled objects, leaving only the environment."""
        self.logger.debug("Cropping environment from point cloud")
        for obj in objects:
            pos, dims, yaw = obj['position'], obj['dimensions'], obj['yaw']
            box = o3d.geometry.OrientedBoundingBox(
                center=np.array(pos),
                R=o3d.geometry.get_rotation_matrix_from_xyz([0, 0, yaw]),
                extent=np.array(dims)
            )
            pc = pc.crop(box, invert=True)
            self.logger.debug(f"Removed object with track_id: {obj['track_id']} from environment")
        self.logger.debug("Environment cropped successfully")
        return pc

    def save_point_cloud(self, pc: o3d.geometry.PointCloud, save_path: str):
        """Save point cloud to file."""
        try:
            o3d.io.write_point_cloud(save_path, pc)
            self.logger.info(f"Point cloud saved to: {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save point cloud to {save_path}: {e}")

    def visualize_point_clouds(self, environment_pc: Optional[o3d.geometry.PointCloud], cropped_pc: Optional[o3d.geometry.PointCloud]):
        """Visualize environment and cropped objects."""
        self.logger.info("Starting point cloud visualization")
        if not environment_pc and not cropped_pc:
            self.logger.warning("No valid point clouds found to display.")
            return

        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            self.logger.debug("Open3D visualization window created")

            if environment_pc:
                environment_pc.paint_uniform_color([0.5, 0.5, 0.5])  # Gray
                vis.add_geometry(environment_pc)
                self.logger.debug("Added environment point cloud to visualization")

            if cropped_pc:
                cropped_pc.paint_uniform_color([1, 0, 0])  # Red
                vis.add_geometry(cropped_pc)
                self.logger.debug("Added cropped objects point cloud to visualization")

            render_option = vis.get_render_option()
            render_option.point_size = self.point_size
            render_option.background_color = np.asarray(self.background_color)

            self.logger.info("Running visualization...")
            vis.run()
            self.logger.info("Visualization finished")
            vis.destroy_window()
            self.logger.debug("Open3D visualization window destroyed")
        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")

    def process_scenario(self):
        """Process the scenario to align the environment and cropped objects."""
        self.logger.info("Starting scenario processing")
        aligned_cropped_pcs = []
        environment_queue = []
        last_positions = {}

        # Pass 1: Identify the last frame where each object appears
        self.logger.info("Starting Pass 1: Identifying last appearances of objects")
        start_frame = self.max_frames - self.alignment_interval
        end_frame = start_frame + self.alignment_interval
        logging.info(f"start_frame: {start_frame}, end_frame: {end_frame}")
        frame_indices_pass1 = list(range(start_frame, end_frame, self.frame_step))
        # frames_to_process_pass1 = min(len(frame_indices_pass1), self.max_frames)

        # with tqdm(total=frames_to_process_pass1, desc="Pass 1: Parsing Labels", unit="frame") as pbar:
        processed_frames = 0
        for frame_index in frame_indices_pass1:
            if processed_frames >= self.max_frames:
                break

            label_file = os.path.join(self.scenario_path, f'label3d_{frame_index:04d}.txt')

            if os.path.isfile(label_file):
                try:
                    objects = self.parse_labels(label_file)
                    for obj in objects:
                        track_id = obj['track_id']
                        last_positions[track_id] = {'frame': frame_index, 'object': obj}
                    processed_frames += 1
                    # pbar.update(1)
                    self.logger.info(f"Processed labels for frame {frame_index}")
                except Exception as e:
                    self.logger.error(f"Error parsing labels from {label_file}: {e}")
            else:
                self.logger.warning(f"Label file does not exist: {label_file}")

        self.logger.info(f"Pass 1 completed: Identified last appearances for {len(last_positions)} objects")

        # Pass 2: Process the frames for alignment
        self.logger.info("Starting Pass 2: Processing frames for alignment")
        start_frame = self.max_frames - self.alignment_interval
        end_frame = start_frame + (self.alignment_interval * 2)
        logging.info(f"start_frame: {start_frame}, end_frame: {end_frame}")
        frame_indices_pass2 = list(range(start_frame, end_frame, self.frame_step))
        # frames_to_process_pass2 = self.max_frames

        # with tqdm(total=frames_to_process_pass2, desc="Pass 2: Aligning Frames", unit="frame") as pbar:
        processed_frames = 0
        for frame_index in frame_indices_pass2:
            if processed_frames >= self.max_frames:
                break

            pc_file = os.path.join(self.scenario_path, f'pc_{frame_index:04d}.ply')
            odom_file = os.path.join(self.scenario_path, f'odom_{frame_index:04d}.txt')
            label_file = os.path.join(self.scenario_path, f'label3d_{frame_index:04d}.txt')

            if not (os.path.isfile(pc_file) and os.path.isfile(odom_file) and os.path.isfile(label_file)):
                self.logger.warning(f"Missing files for frame {frame_index}: PC={os.path.isfile(pc_file)}, Odom={os.path.isfile(odom_file)}, Label={os.path.isfile(label_file)}")
                frame_index += self.frame_step
                continue

            try:
                pc = o3d.io.read_point_cloud(pc_file)
                self.logger.info(f"Loaded point cloud from {pc_file}")

                if self.use_downsampling:
                    pc = pc.voxel_down_sample(voxel_size=self.voxel_size)
                    self.logger.debug(f"Downsampled point cloud with voxel size {self.voxel_size}")
            except Exception as e:
                self.logger.error(f"Failed to read point cloud from {pc_file}: {e}")
                frame_index += self.frame_step
                continue

            try:
                objects = self.parse_labels(label_file)
            except Exception as e:
                self.logger.error(f"Error parsing labels from {label_file}: {e}")
                frame_index += self.frame_step
                continue

            # Select objects appearing in their last frame
            cropped_objects = [
                last_positions[track_id]['object']
                for track_id in last_positions
                if last_positions[track_id]['frame'] == frame_index
            ]
            self.logger.debug(f"Selected {len(cropped_objects)} objects for cropping in frame {frame_index}")

            environment_pc = self.crop_environment(pc, objects)
            cropped_pc = self.crop_objects_from_pcd(pc, cropped_objects)

            try:
                vehicle_odometry = self.load_vehicle_odometry(odom_file)
            except Exception as e:
                self.logger.error(f"Failed to load odometry from {odom_file}: {e}")
                frame_index += self.frame_step
                continue

            if environment_pc:
                try:
                    aligned_env_pc = self.apply_vehicle_transformation(environment_pc, vehicle_odometry)
                    environment_queue.append(aligned_env_pc)
                    if len(environment_queue) > self.alignment_interval:
                        popped_pc = environment_queue.pop(0)
                        self.logger.debug(f"Removed oldest environment point cloud from queue")
                    
                    combined_environment_pc = environment_queue[0]
                    for env_pc in environment_queue[1:]:
                        combined_environment_pc += env_pc
                    self.logger.debug("Combined environment point clouds")
                except Exception as e:
                    self.logger.error(f"Failed to apply transformation to environment point cloud: {e}")

            if cropped_pc:
                try:
                    aligned_cropped_pc = self.apply_vehicle_transformation(cropped_pc, vehicle_odometry)
                    aligned_cropped_pcs.append(aligned_cropped_pc)
                    self.logger.debug("Aligned cropped objects point cloud")
                except Exception as e:
                    self.logger.error(f"Failed to apply transformation to cropped objects point cloud: {e}")

            processed_frames += 1
            # pbar.update(1)

        self.logger.info(f"Pass 2 completed: Processed {processed_frames} frames")

        # Combine all aligned cropped objects into a single point cloud
        self.logger.info("Combining all aligned cropped objects into a single point cloud")
        if aligned_cropped_pcs:
            combined_cropped_pc = aligned_cropped_pcs[0]
            for pc in aligned_cropped_pcs[1:]:
                combined_cropped_pc += pc
            self.logger.debug("Combined cropped point clouds")
        else:
            combined_cropped_pc = None
            self.logger.warning("No cropped point clouds to combine")

        # Merge environment and cropped objects
        self.logger.info("Merging environment and cropped objects point clouds")
        if combined_environment_pc and combined_cropped_pc:
            combined_pcd = combined_environment_pc + combined_cropped_pc
            self.logger.debug("Merged environment and cropped point clouds")
        elif combined_environment_pc:
            combined_pcd = combined_environment_pc
            self.logger.debug("Only environment point cloud present")
        elif combined_cropped_pc:
            combined_pcd = combined_cropped_pc
            self.logger.debug("Only cropped point cloud present")
        else:
            combined_pcd = None
            self.logger.warning("No point clouds to merge")

        # Save combined point clouds if saving is enabled
        if self.save_files and combined_pcd:
            combined_save_path = os.path.join(self.save_directory, "combined_aligned.pcd")
            self.logger.info(f"Saving combined point cloud to {combined_save_path}")
            self.save_point_cloud(combined_pcd, combined_save_path)

        # Visualize the combined point clouds
        self.logger.info("Starting visualization of combined point clouds")
        self.visualize_point_clouds(combined_environment_pc, combined_cropped_pc)
        self.logger.info("Scenario processing completed")


def main():
    args = parse_arguments()
    config = Config()
    aligner = PointCloudOdometryAligner(args, config)
    aligner.process_scenario()


if __name__ == "__main__":
    main()
