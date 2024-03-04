pointcloud_config = dict()
pointcloud_config['base_path'] = "data/pointclouds"
pointcloud_config['voxel_size'] = 0.01
pointcloud_config['nb_neighbors'] = 20
pointcloud_config["std_ratio"] = 2.0
pointcloud_config["radius"] = 0.1
pointcloud_config["max_nn"] = 100

mesh_config = dict()
mesh_config["depth"] = 9
mesh_config["alpha"] = 0.03
mesh_config["radii"] =  [0.005, 0.01, 0.02, 0.04]

pose_config = dict()
pose_config["base_path"] = "data/pointclouds"

rgbd_config = dict()
rgbd_config["base_path"] = "data"