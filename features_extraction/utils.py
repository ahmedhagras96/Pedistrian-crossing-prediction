def parse_odometry(file_path):
    with open(file_path, "r") as f:
        values = list(map(float, f.readline().strip().split(",")))
        return {"x": values[0], "y": values[1], "z": values[2]}

def parse_labels(file_path):
    pedestrians = {}
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if parts[0] == "Pedestrian":
                ped_id = parts[1]
                pos_x = float(parts[3])
                pos_y = float(parts[4])
                pedestrians[ped_id] = {"x": pos_x, "y": pos_y}
    return pedestrians