import open3d as o3d
import numpy as np
import pandas as pd
import os

class ScenarioManager:
    def __init__(self, scenario_name, loki_path=None, NumFrames=0, ped_id=None):
        self.folder = os.path.relpath(os.path.join(os.curdir, scenario_name))
        self.scenario_name = scenario_name
        self.loki_path = loki_path
        self.num_frames = NumFrames
        self.ped_id = ped_id
        self.label_3d = self.get_label3d()
        self.odometry_data = self.get_odometry()
        self.frame_numbers = []
        self.ped_id_to_points = {}
        self.car_id_to_points = {}
        self.ped_id_to_box_info = {}
        self.transformations = None

    def get_label3d(self):
        label_3d = [os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.startswith('label3d')]
        label_3d.sort()
        return label_3d[:self.num_frames][::-1]

    def get_odometry(self):
        odometry_data = {}
        odoms = {}
        for file in os.listdir(self.folder):
            if file.startswith('odom'):
                frame_number = os.path.splitext(file.split('_')[-1])[0]
                odoms[frame_number] = os.path.join(self.folder, file)

        for frame_number, odometry in odoms.items():
            with open(odometry, 'r') as f:
                try:
                    x = [float(i) for i in f.read().split(',') if i.strip()]
                    odometry_data[frame_number] = x
                except ValueError as e:
                    print(f"Error processing odometry file {odometry}: {e}")
        trunc_odometry_data = dict(list(iter(odometry_data.items()))[:self.num_frames])
        return trunc_odometry_data
    
    def process_labels(self):
        for label_file in self.label_3d:
            frame_number = os.path.splitext(os.path.basename(label_file))[0].split('_')[-1]
            frame_path = f'./{self.scenario_name}/pc_{frame_number}.ply'
            frame_point_cloud = o3d.io.read_point_cloud(frame_path)

            transformation_matrix = self.get_transformation_matrix(frame_number)
            data_df = self.parse_label_file(label_file)

            for _, obj_row in data_df.iterrows():
                self.process_object(obj_row, frame_point_cloud, transformation_matrix, frame_number)

    def get_transformation_matrix(self, frame_number):
        transformation_matrix = np.eye(4)
        odom = self.odometry_data.get(frame_number)
        if odom:
            translation = odom[:3]
            rotation = o3d.geometry.get_rotation_matrix_from_xyz(odom[3:6])
            transformation_matrix[:3, :3] = rotation
            transformation_matrix[:3, 3] = translation
        return transformation_matrix

    def parse_label_file(self, label_file):
        with open(label_file, 'r') as f:
            data = [row.split(',') for row in f.read().strip().split('\n')]
        columns = data.pop(0)
        return pd.DataFrame(data, columns=[col.strip() for col in columns])

    def process_object(self, obj_row, frame_point_cloud, transformation_matrix, frame_number):
        if obj_row['labels'] not in ['Pedestrian', 'Car']:
            return

        dimensions = [float(obj_row[col]) for col in obj_row.index[3:10]]
        center_box, yaw, (w, l, h) = dimensions[:3], dimensions[6], dimensions[3:6]
        yaw_matrix = self.get_yaw_matrix(yaw)

        bounding_box_o3d = o3d.geometry.OrientedBoundingBox(center_box, yaw_matrix, [w, l, h])
        points_ix = bounding_box_o3d.get_point_indices_within_bounding_box(frame_point_cloud.points)
        obj_ply = frame_point_cloud.select_by_index(points_ix)
        obj_points = np.asarray(obj_ply.points)
        transformed_points = self.apply_transformation(obj_points, transformation_matrix)
        transformed_center_box = self.apply_transformation(center_box, transformation_matrix).squeeze()

        obj_id = obj_row['track_id']

        if obj_row['labels'] == 'Pedestrian':
            self.store_pedestrian_data(frame_number, obj_id, transformed_points, transformed_center_box, yaw, (w, l, h))
        elif obj_row['labels'] == 'Car':
            self.store_car_data(frame_number, obj_id, transformed_points)

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
        loki_df = pd.read_csv(self.loki_path)
        loki_df_scenario = loki_df[loki_df['video_name'] == self.scenario_name]
        aligned_ped_ID_to_points = {}
        aligned_box_info = {}

        for i, (_, row) in enumerate(loki_df_scenario.iterrows()):
            ped_id = row['Ped_ID']

            if i > self.num_frames: break
            if ped_id != self.ped_id: continue

            frame_name = row['frame_name']
            frame_number = frame_name.split('_')[-1]
            self.frame_numbers.append(frame_number)

            points = self.ped_id_to_points[(frame_number, ped_id)]
            center, yaw, dimensions = self.ped_id_to_box_info[(frame_number, ped_id)]

            if points is None:
                print(f"Missing data for frame {frame_number}, Ped_ID {ped_id}. Skipping...")
                continue

            aligned_ped_ID_to_points.setdefault((ped_id, frame_number), []).append(points)
            aligned_box_info.setdefault((ped_id, frame_number), []).append([center, yaw, dimensions])

        return {
            "aligned_ped_ID_to_points": aligned_ped_ID_to_points,
            "aligned_box_info": aligned_box_info
        }
   
    def bbox_scenario(self, frame_number):   
        # bounding box of the pedestrian in the whole scenario
        aligned_box_info = self.align_pedestrian_data()["aligned_box_info"]
        padding = 20
        center, yaw, dimension = aligned_box_info[self.ped_id, frame_number][0]
        w, l, h = dimension
        w*=padding
        l*=padding
    
        yaw_matrix = self.get_yaw_matrix(yaw)
        dimension = (w,l,h)
        # Create the oriented bounding box for the pedestrian and the environment
        PedEnv_box = o3d.geometry.OrientedBoundingBox(center, yaw_matrix, dimension)

        return PedEnv_box
    
    def crop_bbox(self, frame_number):

        PedEnv_box = self.bbox_scenario(frame_number)
        ped_data = self.align_pedestrian_data()["aligned_ped_ID_to_points"][self.ped_id, frame_number][0]

        car_points = [np.concatenate(points, axis=0) for points in self.car_id_to_points.get(frame_number, {}).values()]

        map_path = os.path.join((os.path.relpath(self.scenario_name)), "map.ply")
        map_points = np.asarray(o3d.io.read_point_cloud(map_path).points)

        scenario_points = np.concatenate([ped_data, car_points[0], map_points], axis=0)
        scenario_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scenario_points))

        # crop the points in the PedEnv_box 
        points_ix = PedEnv_box.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(scenario_points))
        scenario_selected_pcd = scenario_pcd.select_by_index(points_ix)

        return scenario_selected_pcd



    def visualize(self, cropped = True):

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().background_color = np.array([0.2, 0.2, 0.2])  
        vis.get_render_option().point_size = 2.0  

        last_frame = self.frame_numbers[-1]

        if cropped:
            scenario_pcd = self.crop_bbox(last_frame)
            vis.add_geometry(scenario_pcd)
        else:

            # Adding a single pedestrain based on its ID
            ped_data = self.align_pedestrian_data()["aligned_ped_ID_to_points"][self.ped_id, last_frame][0]
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
            map_path = os.path.join((os.path.relpath(self.scenario_name)), "map.ply")
            map_pcd = o3d.io.read_point_cloud(map_path)
            vis.add_geometry(map_pcd)
    
        print(f'visualizing {last_frame} frame')

        vis.run()
        vis.destroy_window()

        

if __name__ == "__main__":
    processor = ScenarioManager(scenario_name="scenario_026", loki_path="./loki.csv", NumFrames= 40, ped_id='624e3a59-7b6f-4674-a223-41966cdfa39a')
    processor.process_labels()
    pedestrian_data = processor.align_pedestrian_data() 
    processor.visualize(cropped=False)





    