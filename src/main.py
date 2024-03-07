from utils.data_utils import get_files, get_paths, get_pcd_path, get_pose_path, generate_camera_trajectory
from config.config import pointcloud_config, mesh_config, pose_config, rgbd_config
from reconstruction import PointCloud, MeshReconstruction
from pipeline import MultiwayRegistration, CombinePointclouds
import argparse

parser = argparse.ArgumentParser(
    prog = "Mesh Reconstruction",
    description="The python package performs Mesh Reconstruction from given pointcloud"
)

parser.add_argument('-d', '--data_type')
parser.add_argument("-s", '--scene_type')
parser.add_argument("-r", '--recon_method')
args = parser.parse_args()

if __name__ == "__main__":
    result_pcd = f'results/{args.data_type}_{args.scene_type}.pcd'
    if(args.data_type == "direct"):
        params = dict()
        params["pcd_path"] = get_pcd_path(pointcloud_config["base_path"], 
                                          args.scene_type)
        p = PointCloud(args.data_type, params)
        pcd = p.read_pointcloud()
        pcd = p.downsample_pointcloud(pcd, pointcloud_config["voxel_size"])
        pcd, ind = p.statistical_outlier_removal(pcd, 
                                                 pointcloud_config["nb_neighbors"],
                                                 pointcloud_config["std_ratio"])
        pcds = [pcd]
        p.visualize_inliers_outliers(ind, pcd)
        p.write_pointcloud(pcd, result_pcd)

    elif(args.data_type == "registration"):
        rgb_image_paths = get_files("data")
        m = MultiwayRegistration(rgb_image_paths, pointcloud_config["voxel_size"])
        pose_graph, pcds = m.optimize_posegraph(image_set = args.scene_type)
        m.visualize_combined_pointclouds(pcds, pose_graph)
        pcd = m.merge_pointcloud(pcds, pose_graph)
        print(pcd)
        m.write_pointcloud(pcd, result_pcd)
    
    else:
        params = dict()
        pose_file_path, trajectory_file_path = get_pose_path(pose_config["base_path"], args.scene_type)
        generate_camera_trajectory(pose_file_path, trajectory_file_path)
        rgb_image_paths = get_files(rgbd_config["base_path"])
        c = CombinePointclouds(rgb_image_paths[args.scene_type], trajectory_file_path)
        pcds = c.combine_pointclouds()
        c.visualize(pcds)
        pcd = c.merge_pointcloud(pcds)
        print(pcd)
        c.write_pointcloud(pcd, result_pcd)
    
    # result_mesh = f'results/{args.data_type}_{args.scene_type}_{args.recon_method}.obj'
    # me = MeshReconstruction()
    # if (args.recon_method == "alpha"):
    #     mesh = me.alpha_shapes(pcd, mesh_config["alpha"])
    # elif (args.recon_method == "ball_pivot"):
    #     mesh = me.ball_pivot(pcd, mesh_config["radii"])
    # else:
    #     mesh = me.poisson(pcd, mesh_config["depth"])
    
    # me.visualize_mesh(mesh)
    # me.write_mesh(mesh, result_mesh)

