import open3d as o3d
import numpy as np
import pandas as pd
import os
import gc
import logging

# Set up logging with color-coded levels
class ColorFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: "\033[92m",   # Green
        logging.INFO: "\033[94m",    # Blue
        logging.WARNING: "\033[93m", # Yellow
        logging.ERROR: "\033[91m",   # Red
        logging.CRITICAL: "\033[1;91m", # Bright Red
    }
    RESET_COLOR = "\033[0m"

    def format(self, record):
        level_color = self.LEVEL_COLORS.get(record.levelno, self.RESET_COLOR)
        message = super().format(record)
        return f"{level_color}{message}{self.RESET_COLOR}"

# Configure logging
handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(level=logging.DEBUG, handlers=[handler])

logger = logging.getLogger(__name__)
class ScenarioManager:
    def __init__(self, scenario_name, loki_path=None, NumFrames=0, ped_id=None):
        self.folder_path = os.path.normpath(os.path.join(os.path.dirname((__file__)), '../LOKI/'))
        self.scenario_path = os.path.join(self.folder_path, scenario_name)
        self.loki_path = os.path.join(self.folder_path, loki_path)
        self.map_path = os.path.join(self.scenario_path, 'map.ply')
        self.scenario_name = scenario_name
        self.num_frames = NumFrames
        self.ped_id = ped_id
        self.frame_numbers = self.get_relevant_frames()
        self.label_3d = self.get_label3d()
        self.odometry_data = self.get_odometry()
        self.ped_id_to_points = {}
        self.car_id_to_points = {}
        self.ped_id_to_box_info = {}
        self.aligned_ped_ID_to_points = {}
        self.aligned_box_info = {}
        self.transformations = None
    
    def get_relevant_frames(self):
        loki_df = pd.read_csv(self.loki_path)
        loki_df_scenario = loki_df[loki_df['video_name'] == self.scenario_name]

        frame_numbers = []
        i = 0
        for _, row in loki_df_scenario.iterrows():
            ped_id = row['Ped_ID']
            frame_name = row['frame_name']
            frame_number = frame_name.split('_')[-1]
            
            if ped_id != self.ped_id: 
                continue
            else: 
                i += 1

            if i > self.num_frames: break

            frame_numbers.append(frame_number)
        return frame_numbers

    def get_label3d(self):
        assert self.num_frames > 0, "Number of frames must be greater than 0"
        label_3d = []
        # populate label_3d based on frame numbers
        for i ,frame_number in enumerate(self.frame_numbers):
            label_path = os.path.join(self.scenario_path, f"label3d_{frame_number}.txt")
            label_3d.append(label_path)
        logger.info(f"Loaded label3d for {len(label_3d)} frames.")
        return label_3d

    def get_odometry(self):
        odometry_data = {}
        odometry = {os.path.splitext(file.split('_')[-1])[0]: os.path.join(self.scenario_path, file)
                    for file in os.listdir(self.scenario_path) if file.startswith('odom')}
        # Load odometry only for relevant frames
        for frame_number in self.frame_numbers:
            odometry_path = odometry.get(frame_number)
            if not odometry_path:
                logger.warning(f"Missing odometry file for frame {frame_number}.")
                continue

            try:
                with open(odometry_path, 'r') as f:
                    odometry_data[frame_number] = [float(i) for i in f.read().split(',') if i.strip()]
            except ValueError as e:
                logger.error(f"Error reading odometry file {odometry_path}: {e}")

        logger.info(f"Loaded odometry for {len(odometry_data)} frames.")
        return odometry_data

    def process_labels(self):
        for label_file in self.label_3d:
            frame_number = os.path.splitext(os.path.basename(label_file))[0].split('_')[-1]
            frame_path = os.path.join(self.folder_path, self.scenario_name, f'pc_{frame_number}.ply')
            frame_point_cloud = o3d.io.read_point_cloud(frame_path)

            if len(frame_point_cloud.points) == 0:
                logger.warning(f"Point cloud for frame {frame_number} is empty.")
                continue

            logger.info(f"Processing point cloud for frame {frame_number} with {len(frame_point_cloud.points)} points.")

            transformation_matrix = self.get_transformation_matrix(frame_number)
            data_df = self.parse_label_file(label_file)

            for _, obj_row in data_df.iterrows():
                self.process_object(obj_row, frame_point_cloud, transformation_matrix, frame_number)
            
            # Release memory
            del frame_point_cloud, data_df
            gc.collect()

    def get_transformation_matrix(self, frame_number):
        transformation_matrix = np.eye(4)
        odom = self.odometry_data.get(frame_number)
        if odom:
            translation = odom[:3]
            rotation = o3d.geometry.get_rotation_matrix_from_xyz(odom[3:6])
            transformation_matrix[:3, :3] = rotation
            transformation_matrix[:3, 3] = translation
        logger.debug(f"Transformation matrix for frame {frame_number}: {transformation_matrix}")
        return transformation_matrix

    def parse_label_file(self, label_file):
        try:
            with open(label_file, 'r') as f:
                data = [row.split(',') for row in f.read().strip().split('\n')]
            columns = data.pop(0)
            return pd.DataFrame(data, columns=[col.strip() for col in columns])
        except Exception as e:
            logger.error(f"Error reading label file {label_file}: {e}")
            return pd.DataFrame()
        
    def process_object(self, obj_row, frame_point_cloud, transformation_matrix, frame_number):
        try:
            if obj_row['labels'] not in ['Pedestrian', 'Car']:
                logger.debug(f"Skipping non-relevant object (label={obj_row['labels']}) in frame {frame_number}.")
                return

            # Check if the pedestrian ID matches for relevant pedestrian objects
            if obj_row['labels'] == 'Pedestrian':
                if obj_row.get('track_id') != self.ped_id:
                    logger.debug(
                        f"Skipping pedestrian with track_id={obj_row['track_id']} in frame {frame_number}, "
                        f"does not match target ped_id={self.ped_id}."
                    )
                    return
                
            # Log relevant car objects (optional, if needed for debugging)
            if obj_row['labels'] == 'Car':
                logger.debug(f"Processing car object with track_id={obj_row['track_id']} in frame {frame_number}.")

            center_box = [float(obj_row[col]) for col in ['pos_x', 'pos_y', 'pos_z']]
            dimensions = [float(obj_row[col]) for col in ['dim_x', 'dim_y', 'dim_z']]
            yaw = float(obj_row['yaw'])
            
            # Check if dimensions are valid
            if any(dim <= 0 for dim in dimensions):
                logger.warning(f"Invalid dimensions for bounding box in frame {frame_number}: {dimensions}")
                return
            
            bounding_box_o3d = o3d.geometry.OrientedBoundingBox(center_box, self.get_yaw_matrix(yaw), dimensions)
            logger.debug(f"Bounding box created for frame {frame_number}: center={center_box}, dimensions={dimensions}, yaw={yaw}")
            
            points_ix = bounding_box_o3d.get_point_indices_within_bounding_box(frame_point_cloud.points)
            if not points_ix:
                logger.warning(f"No points found within bounding box for frame {frame_number}. For {obj_row['labels']} with track_id={obj_row['track_id']}")
                
            
            obj_ply = frame_point_cloud.select_by_index(points_ix)
            obj_points = np.asarray(obj_ply.points)
            transformed_points = self.apply_transformation(obj_points, transformation_matrix)
            transformed_center_box = self.apply_transformation(center_box, transformation_matrix).squeeze()

            obj_id = obj_row['track_id']

            if obj_row['labels'] == 'Pedestrian':
                self.store_pedestrian_data(frame_number, obj_id, transformed_points, transformed_center_box, yaw, dimensions)
            elif obj_row['labels'] == 'Car':
                self.store_car_data(frame_number, obj_id, transformed_points)

            logger.info(f"Object processed with {len(transformed_points)} transformed points for frame {frame_number}.")
        except Exception as e:
            logger.error(f"Error processing object in frame {frame_number}: {e}")

    def get_yaw_matrix(self, yaw):
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        return np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])

    def apply_transformation(self, points, transformation_matrix):
        points = np.asarray(points)
        if points.ndim == 1:  # Single point case
            points = points.reshape(1, -1)
        elif points.shape[1] != 3:
            raise ValueError(f"Unexpected shape for points: {points.shape}. Expected (N, 3).")

        return np.dot(np.hstack((points, np.ones((points.shape[0], 1)))), transformation_matrix.T)[:, :3]

    def store_pedestrian_data(self, frame_number, obj_id, points, center_box, yaw, dimensions):
        self.ped_id_to_points[(frame_number, obj_id)] = points
        self.ped_id_to_box_info[(frame_number, obj_id)] = (center_box, yaw, dimensions)

    def store_car_data(self, frame_number, obj_id, points):
        self.car_id_to_points.setdefault(frame_number, {}).setdefault(obj_id, []).append(points)
    

    def align_pedestrian_data(self):
        for frame_number in self.frame_numbers:
            ped_id = self.ped_id            

            points = self.ped_id_to_points[(frame_number, ped_id)]
            center, yaw, dimensions = self.ped_id_to_box_info[(frame_number, ped_id)]

            if len(points) == 0:
                logging.warning(f"Pedestrian with ID {ped_id} has no points for frame {frame_number}.")
                logging.warning(f'Pedestrian points will not be visualized for frame {frame_number}.')
                
            self.aligned_ped_ID_to_points.setdefault((ped_id, frame_number), []).append(points)
            self.aligned_box_info.setdefault((ped_id, frame_number), []).append([center, yaw, dimensions])

   
    def bbox_scenario(self, frame_number):   
        # bounding box of the pedestrian in the whole scenario
        padding = 20
        center, yaw, dimension = self.aligned_box_info[self.ped_id, frame_number][0]
        w, l, h = dimension
        w*=padding
        l*=padding
        h*=padding
    
        yaw_matrix = self.get_yaw_matrix(yaw)
        dimension = (w,l,h)
        # Create the oriented bounding box for the pedestrian and the environment
        PedEnv_box = o3d.geometry.OrientedBoundingBox(center, yaw_matrix, dimension)

        return PedEnv_box
    
    def crop_bbox(self, pedestrian_points, frame_number):

        PedEnv_box = self.bbox_scenario(frame_number)

        car_points = [np.concatenate(points, axis=0) for points in self.car_id_to_points.get(frame_number, {}).values()]

        map_points = np.asarray(o3d.io.read_point_cloud(self.map_path).points)

        scenario_points = np.concatenate([pedestrian_points, car_points[0], map_points], axis=0)
        scenario_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scenario_points))

        # crop the points in the PedEnv_box 
        points_ix = PedEnv_box.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(scenario_points))
        scenario_selected_pcd = scenario_pcd.select_by_index(points_ix)
        
        # Clean up
        del scenario_pcd, scenario_points, points_ix
        gc.collect()

        return scenario_selected_pcd



    def visualize(self, cropped = True):

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().background_color = np.array([0.2, 0.2, 0.2])  # Set background color
        vis.get_render_option().point_size = 2.0  # Set point size
        last_frame = self.frame_numbers[-1]
        ped_data = self.aligned_ped_ID_to_points[self.ped_id, last_frame][0]

        if cropped:
            scenario_pcd = self.crop_bbox(ped_data, last_frame)
            vis.add_geometry(scenario_pcd)
        else:

            # Adding a single pedestrain based on its ID
            ped_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ped_data))
            ped_pcd.paint_uniform_color([0.5, 0.5, 1])
            vis.add_geometry(ped_pcd)
        
            # Adding the bounding box of the pedestrian
            PedEnv_box = self.bbox_scenario(last_frame)
            vis.add_geometry(PedEnv_box)

            # Adding all the cars within a specific frame
            for car_id, points in self.car_id_to_points.get(last_frame, {}).items():
                car_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(points, axis=0)))
                car_pcd.paint_uniform_color([1, 0.5, 0.5])  
                vis.add_geometry(car_pcd)

            # Adding the map
            map_pcd = o3d.io.read_point_cloud(self.map_path)
            vis.add_geometry(map_pcd)
       
        print(f"{self.frame_numbers}\nframes loaded within the range {self.num_frames} for pedestrian {self.ped_id}")
        print(f'visualizing {last_frame} frame')

        gc.collect()
        vis.run()
        vis.destroy_window()

        

if __name__ == "__main__":
    processor = ScenarioManager(scenario_name="scenario_026", loki_path="loki.csv", NumFrames=20, ped_id='4ff8af4d-6840-47c2-bc9b-eb383009ad65')
    processor.process_labels()
    processor.align_pedestrian_data() 
    processor.visualize(cropped=False)





    