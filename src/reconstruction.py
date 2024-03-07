from open3d import io, geometry, camera, visualization, utility
from utils.data_utils import get_paths
import numpy as np
import matplotlib.pyplot as plt

class CameraPose:

    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat

    def __str__(self):
        return 'Metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
            "Pose : " + "\n" + np.array_str(self.pose)

class RGBD():
    def __init__(self, rgb_image_path, depth_image_path):
        self.rgb_image = io.read_image(rgb_image_path)
        self.depth_image = io.read_image(depth_image_path)
        return

    def create_rgbd(self):
        self.rgbd_image = \
            geometry.\
            RGBDImage. \
                create_from_color_and_depth(self.rgb_image, self.depth_image, convert_rgb_to_intensity=False)
        return
    
    def get_rgbd(self):
        self.create_rgbd()
        return self.rgbd_image
    
class PointCloud(RGBD):
    def __init__(self, data_type, params):
        if(data_type == "RGBD"):
            super().__init__(params["rgb_image_path"], params["depth_image_path"])
            self.data_type = data_type
        else:
            self.data_type = data_type
            self.pcd_path = params["pcd_path"]
        return
    
    def read_pointcloud(self):
        pcd = io.read_point_cloud(self.pcd_path)
        return pcd
    
    def create_pointcloud_from_rgbd(self, intrinsic = camera.PinholeCameraIntrinsic(
        camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault), extrinsic=None):
        rgbd_image = self.get_rgbd()
        pcd = geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,intrinsic, extrinsic
        )
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        return pcd
    
    def downsample_pointcloud(self,pcd, voxel_size=0.01):
        downpcd = pcd.voxel_down_sample(voxel_size)
        return downpcd
    
    def statistical_outlier_removal(self, pcd, nb_neighbors=20, std_ratio=2.0):
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors,
                                                    std_ratio)
        return pcd, ind
    
    def estimate_normals_in_pointcloud(self, pcd, radius=0.1, max_nn=100):
        pcd.estimate_normals(
        search_param= geometry.KDTreeSearchParamHybrid(radius, max_nn))
        pcd.orient_normals_consistent_tangent_plane(100)
        return pcd
    
    def visualize_images(self):
        rgbd_image = self.get_rgbd()
        plt.subplot(1, 2, 1)
        plt.title('Image: ')
        plt.imshow(rgbd_image.color)
        plt.subplot(1, 2, 2)
        plt.title('Depth Image: ')
        plt.imshow(rgbd_image.depth)
        plt.show()
        return
    
    def visualize_pointcloud(self, pcd):
        visualization.draw_geometries([pcd])
        return
    
    def visualize_inliers_outliers(self, ind, pcd):
        inlier_cloud = pcd.select_by_index(ind)
        outlier_cloud = pcd.select_by_index(ind, invert=True)

        print("Showing outliers (red) and inliers (gray): ")
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        visualization.draw_geometries([inlier_cloud, outlier_cloud])

    def merge_pointcloud(self, pcds):
        merged_pcd = geometry.PointCloud()
        for pcd in pcds:
            merged_pcd += pcd
        return merged_pcd
    
    def write_pointcloud(self, pcds, pcd_path):
        io.write_point_cloud(pcd_path, pcds)
        return
    
class MeshReconstruction():
    def __init__(self):
        return
    
    def alpha_shapes(self, pcd, alpha=0.03):
        mesh = geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        mesh.compute_vertex_normals()
        return mesh
    
    def ball_pivot(self, pcd, radii):
        mesh = geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, utility.DoubleVector(radii))
        return mesh

    def poisson(self, pcd, depth=9):
        with utility.VerbosityContextManager(
        utility.VerbosityLevel.Debug) as cm:
            mesh, densities = geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth)
            vertices_to_remove = densities < np.quantile(densities, 0.01)
            mesh.remove_vertices_by_mask(vertices_to_remove)
        return mesh
    
    def visualize_mesh(self, mesh):
        visualization.draw_geometries([mesh], mesh_show_back_face=True)
        return

    def write_mesh(self, mesh, mesh_path):
        io.write_triangle_mesh(mesh_path, mesh)
        return




