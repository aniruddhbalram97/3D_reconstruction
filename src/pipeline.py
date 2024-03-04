from reconstruction import PointCloud, RGBD
from open3d import io, geometry, camera, visualization, utility, pipelines
from utils.data_utils import get_paths, get_files
import numpy as np

class MultiwayRegistration():
    def __init__(self, rgb_image_paths, voxel_size):
        self.voxel_size = voxel_size
        self.rgb_image_paths = rgb_image_paths
        self.max_correspondence_distance_coarse = voxel_size * 15
        self.max_correspondence_distance_fine = voxel_size * 1.5
        return
    
    def get_pointclouds(self, image_set):
        pcds = []
        for rgb_image_path in self.rgb_image_paths[image_set]:
            try:
                print(f"Generating Pointcloud for: {rgb_image_path}\n")
                _, depth_image_path = get_paths(rgb_image_path)
                params = dict()
                params["rgb_image_path"], params["depth_image_path"] = rgb_image_path, depth_image_path
                p = PointCloud("RGBD", params)
                pcd = p.create_pointcloud_from_rgbd(extrinsic=np.eye(4))
                pcd = p.downsample_pointcloud(pcd, self.voxel_size)
                pcd, ind = p.statistical_outlier_removal(pcd)
                pcd = p.estimate_normals_in_pointcloud(pcd.select_by_index(ind))
                pcds.append(pcd)
            except Exception as e:
                print("Couldn't create pointcloud...skipping image")
                
        return pcds
    
    def pairwise_registration(self, source_pcd, target_pcd):
        print("Apply point-to-plane ICP")
        icp_coarse = pipelines.registration.registration_icp(
            source_pcd, target_pcd, self.max_correspondence_distance_coarse, np.identity(4),
            pipelines.registration.TransformationEstimationPointToPlane())
        icp_fine = pipelines.registration.registration_icp(
            source_pcd, target_pcd, self.max_correspondence_distance_fine,
            icp_coarse.transformation,
            pipelines.registration.TransformationEstimationPointToPlane())
        transformation_icp = icp_fine.transformation
        information_icp = pipelines.registration.get_information_matrix_from_point_clouds(
            source_pcd, target_pcd, self.max_correspondence_distance_fine,
            icp_fine.transformation)
        return transformation_icp, information_icp
    
    def full_registration(self, image_set='background_1'):
        pose_graph = pipelines.registration.PoseGraph()
        odometry = np.identity(4)
        pose_graph.nodes.append(pipelines.registration.PoseGraphNode(odometry))
        pcds = self.get_pointclouds(image_set)
        n_pcds = len(pcds)
        for source_id in range(n_pcds):
            for target_id in range(source_id + 1, n_pcds):
                transformation_icp, information_icp = self.pairwise_registration(
                    pcds[source_id], pcds[target_id])
                print("Build o3d.pipelines.registration.PoseGraph")
                if target_id == source_id + 1:  # odometry case
                    odometry = np.dot(transformation_icp, odometry)
                    pose_graph.nodes.append(
                        pipelines.registration.PoseGraphNode(
                            np.linalg.inv(odometry)))
                    pose_graph.edges.append(
                        pipelines.registration.PoseGraphEdge(source_id,
                                                                target_id,
                                                                transformation_icp,
                                                                information_icp,
                                                                uncertain=False))
                else:  # loop closure case
                    pose_graph.edges.append(
                        pipelines.registration.PoseGraphEdge(source_id,
                                                                target_id,
                                                                transformation_icp,
                                                                information_icp,
                                                                uncertain=True))
        return pose_graph, pcds
    
    def get_pose_graph(self, image_set="background_1"):
        with utility.VerbosityContextManager(
            utility.VerbosityLevel.Debug) as cm:
            pose_graph, pcds = self.full_registration(image_set)
        return pose_graph, pcds
    
    def optimize_posegraph(self, image_set="background_1"):
        print("Optimizing PoseGraph ...")
        option = pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=self.max_correspondence_distance_fine,
            edge_prune_threshold=0.25,
            reference_node=0)
        pose_graph, pcds = self.get_pose_graph(image_set)
        convergence_criteria = pipelines.registration.GlobalOptimizationConvergenceCriteria()
        convergence_criteria.max_iteration = 1000
        convergence_criteria.max_iteration_lm = 1000
        with utility.VerbosityContextManager(
                utility.VerbosityLevel.Debug) as cm:
            pipelines.registration.global_optimization(
                pose_graph,
                pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                convergence_criteria,
                option)
        return pose_graph, pcds
            
    def visualize_combined_pointclouds(self,pcds,pose_graph):
        for point_id in range(len(pcds)):
            print(pose_graph.nodes[point_id].pose)
            pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        visualization.draw_geometries(pcds)
        return

    def merge_pointcloud(self, pcds, pose_graph):
        merged_pcd = geometry.PointCloud()
        for point_id in range(len(pcds)):
            pcds[point_id].transform(pose_graph.nodes[point_id].pose)
            merged_pcd += pcds[point_id]
        return merged_pcd
    
    def write_pointcloud(self, pcds, pcd_path):
        io.write_point_cloud(pcd_path, pcds)
        return
    
class CombinePointclouds():
    def __init__(self, rgb_image_paths, trajectory_file):
        self.trajectory_file = trajectory_file
        self.rgb_image_paths = rgb_image_paths
        return
    
    def read_camera_trajectory(self):
        camera_trajectory = io.read_pinhole_camera_trajectory(self.trajectory_file)
        return camera_trajectory
    
    def combine_pointclouds(self):
        camera_trajectory = self.read_camera_trajectory()
        pcds = []
        for i in range(0, len(camera_trajectory.parameters)):
            print("Generating Pointcloud for: ", self.rgb_image_paths[i])
            _, depth_image_path = get_paths(self.rgb_image_paths[i])
            params = dict()
            params["rgb_image_path"], params["depth_image_path"] = self.rgb_image_paths[i], depth_image_path
            p = PointCloud("RGBD", params)
            intrinsic = camera_trajectory.parameters[i].intrinsic
            extrinsic = camera_trajectory.parameters[i].extrinsic
            pcd = p.create_pointcloud_from_rgbd(intrinsic, extrinsic)
            pcd = p.downsample_pointcloud(pcd)
            pcd, ind = p.statistical_outlier_removal(pcd)
            pcd = p.estimate_normals_in_pointcloud(pcd.select_by_index(ind))
            pcds.append(pcd)
        return pcds
    
    def merge_pointcloud(self, pcds):
        merged_pcd = geometry.PointCloud()
        for pcd in pcds:
            merged_pcd += pcd
        return merged_pcd
    
    def write_pointcloud(self, pcds, pcd_path):
        io.write_point_cloud(pcd_path, pcds)
        return
    
    def visualize(self, pcds):
        visualization.draw_geometries(pcds)
        return