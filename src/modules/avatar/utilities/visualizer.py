import open3d as o3d

from modules.avatar.objects.bounding_box import BoundingBox


class Visualizer:
    """
    Handles visualization of point clouds and bounding boxes using Open3D.
    """

    def __init__(self):
        """
        Initializes the Visualizer.
        """
        self.geometries = []
        self.bounding_boxes = []  # To store bounding box geometries
        self.pedestrian_pcds = []  # To store cropped pedestrian point clouds

    def add_point_cloud(self, pcd):
        """
        Adds a point cloud to the visualization.

        Args:
            pcd (o3d.geometry.PointCloud): The point cloud object.
        """
        self.geometries.append(pcd)

    def add_bounding_boxes(self, df_pedestrians, color=None):
        """
        Adds pedestrian bounding boxes to the visualization.

        Args:
            df_pedestrians (pd.DataFrame): DataFrame containing pedestrian bounding box information.
            color (list, optional): RGB color for the bounding boxes. Defaults to Blue ([0, 0, 1]).
        """
        if color is None:
            color = [0, 0, 1]

        for idx, row in df_pedestrians.iterrows():
            bbox_obj = BoundingBox(row)
            obb = bbox_obj.obb
            if obb is not None:
                obb.color = color
                self.geometries.append(obb)
                self.bounding_boxes.append(obb)  # Keep track of bounding boxes separately

    def clear_bounding_boxes(self):
        """
        Removes all bounding boxes from the visualization.
        """
        if not self.bounding_boxes:
            print("No bounding boxes to remove.")
            return

        for bbox in self.bounding_boxes:
            if bbox in self.geometries:
                self.geometries.remove(bbox)
        self.bounding_boxes = []  # Clear the list after removal
        print("All bounding boxes have been cleared.")

    def add_coordinate_axes(self, size=5.0, origin=None):
        """
        Adds coordinate axes to the visualization.

        Args:
            size (float, optional): Size of the coordinate frame. Defaults to 5.0.
            origin (list, optional): Origin point of the coordinate frame. Defaults to [0, 0, 0].
        """
        if origin is None:
            origin = [0, 0, 0]

        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)
        self.geometries.append(axes)

    def extract_pedestrian_pcds(self, pcd, df_pedestrians):
        """
        Extracts pedestrian-specific point clouds by cropping the original point cloud
        based on the bounding boxes.

        Args:
            pcd (o3d.geometry.PointCloud): The original point cloud.
            df_pedestrians (pd.DataFrame): DataFrame containing pedestrian bounding box information.

        Returns:
            list of o3d.geometry.PointCloud: List of pedestrian-specific point clouds.
        """
        pedestrian_pcds = []
        for idx, row in df_pedestrians.iterrows():
            bbox = BoundingBox(row).obb
            if bbox is not None:
                # Use the 'crop' method of PointCloud
                cropped_pcd = pcd.crop(bbox)
                pedestrian_pcds.append(cropped_pcd)
        self.pedestrian_pcds = pedestrian_pcds
        return pedestrian_pcds

    def visualize(self, window_name='Visualization', width=1280, height=720):
        """
        Launches the Open3D visualization window with all geometries.

        Args:
            window_name (str, optional): Title of the visualization window. Defaults to 'Visualization'.
            width (int, optional): Width of the window. Defaults to 1280.
            height (int, optional): Height of the window. Defaults to 720.
        """
        o3d.visualization.draw_geometries(
            self.geometries,
            window_name=window_name,
            width=width,
            height=height,
            left=50,
            top=50,
            point_show_normal=False,
            mesh_show_wireframe=False,
            mesh_show_back_face=False
        )

    def visualize_pedestrians_only(self, window_name: object = 'Pedestrians Only', width: object = 1280,
                                   height: object = 720,
                                   color=None) -> None:
        """
        Visualizes only the pedestrian point clouds.

        Args:
            window_name (str, optional): Title of the visualization window. Defaults to 'Pedestrians Only'.
            width (int, optional): Width of the window. Defaults to 1280.
            height (int, optional): Height of the window. Defaults to 720.
            color (list, optional): RGB color to assign to all pedestrian point clouds. Defaults to Red ([1, 0, 0]).
        """
        if color is None:
            color = [1, 0, 0]

        pedestrian_geometries = []
        for pcd in self.pedestrian_pcds:
            # Optionally, assign a color to the pedestrian pcd
            colored_pcd = pcd.paint_uniform_color(color)
            pedestrian_geometries.append(colored_pcd)

        # Add coordinate axes
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
        pedestrian_geometries.append(axes)

        # Visualize
        o3d.visualization.draw_geometries(
            pedestrian_geometries,
            window_name=window_name,
            width=width,
            height=height,
            left=50,
            top=50,
            point_show_normal=False,
            mesh_show_wireframe=False,
            mesh_show_back_face=False
        )

    def clear_geometries(self):
        """
        Clears all geometries from the visualizer.
        """
        self.geometries = []
        self.bounding_boxes = []
        self.pedestrian_pcds = []
        print("Visualizer geometries cleared.")
