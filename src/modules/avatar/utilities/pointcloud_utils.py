import open3d as o3d

def convert_from_vertex_to_open3d_pcd(vertex_data):
    """
    Converts vertex data to an Open3D PointCloud object.

    Args:
        vertex_data (numpy.ndarray): Array of vertex data with at least 3 columns (x, y, z).

    Returns:
        o3d.geometry.PointCloud: Converted Open3D PointCloud object.
    """
    if vertex_data.ndim != 2 or vertex_data.shape[1] < 3:
        raise ValueError("vertex_data must be a 2D NumPy array with at least 3 columns for x, y, z coordinates.")

    # Extract XYZ coordinates
    xyz = vertex_data[:, :3]

    # Initialize Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # If colors are available (assuming next three columns), assign them
    if vertex_data.shape[1] >= 6:
        colors = vertex_data[:, 3:6] / 255.0  # Normalize if colors are in [0, 255]
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd