from open3d import io, visualization
from reconstruction import PointCloud, MeshReconstruction
import numpy as np
import argparse
from config.config import mesh_config
import time

parser = argparse.ArgumentParser(
    prog = "Mesh Reconstruction",
    description="The python package performs Mesh Reconstruction from given pointcloud"
)

parser.add_argument('-d', '--data_type')
parser.add_argument("-s", '--scene_type')
parser.add_argument("-r", '--recon_method')

args = args = parser.parse_args()
if __name__ == "__main__":
    ###### MESH QUALITY #######
    # mesh = io.read_triangle_mesh("results/desk_1_poisson_poisson_12.obj")
    # print("triangles: ", mesh.triangles)
    # print("vertices: ", mesh.vertices)
    # print("watertight: ", mesh.is_watertight())
    # print("self-intersecting: ", mesh.is_self_intersecting())
    ###### MESH RECONSTRUCTION/ Demo #########
    params2 = dict()
    params2["pcd_path"]=f"results/registration_{args.scene_type}.pcd"
    p2 = PointCloud("pointcloud", params2)
    pcd = p2.read_pointcloud()
    pcd = p2.downsample_pointcloud(pcd)
    pcd, ind = p2.statistical_outlier_removal(pcd)
    p2.visualize_pointcloud(pcd)
    result_mesh = f'results/{args.scene_type}_{args.recon_method}.obj'
    me = MeshReconstruction()
    if (args.recon_method == "alpha"):
        alpha = np.log10(0.01)
        time_start = time.time()
        print("Mesh reconstruction with Alpha Shapes started at: ", time_start)
        result_mesh = f'results/{args.scene_type}_{args.recon_method}_{alpha}.obj'
        mesh = me.alpha_shapes(pcd, alpha)
        time_end = time.time()
        print("Mesh reconstruction with Alpha Shapes ended at: ", time_end)
        print(f"Total Time Taken for {alpha}!", time_end - time_start)
        me.visualize_mesh(mesh)
        # me.write_mesh(mesh, result_mesh)
    elif (args.recon_method == "ball_pivot"):
        # array_of_radii = [[0.005, 0.01, 0.02, 0.04],[0.01, 0.02, 0.04, 0.08], [0.015, 0.03, 0.06, 0.12], [0.020, 0.04, 0.08, 0.16]]
        radii = [0.020, 0.04, 0.08, 0.16]
        # for idx, radii in enumerate(array_of_radii):
        result_mesh = f'results/{args.scene_type}_{args.recon_method}_ball_pivot_{4}.obj'
        time_start = time.time()
        print("Mesh reconstruction with Ball Pivot started at: ", time_start)
        print("Estimating normals...")
        pcd = p2.estimate_normals_in_pointcloud(pcd)
        print("Time taken for normal estimation: ", time.time() - time_start)
        mesh = me.ball_pivot(pcd, radii)
        time_end = time.time()
        print("Mesh reconstruction with Ball Pivot ended at: ", time_end)
        print("Total Time Taken!", time_end - time_start)
        me.visualize_mesh(mesh)
        # me.write_mesh(mesh, result_mesh)
    else:
        # depths = [9, 10, 11, 12]
        depth = 12
        # for depth in depths:
        result_mesh = f'results/{args.scene_type}_{args.recon_method}_poisson_{depth}.obj'
        time_start = time.time()
        print("Mesh reconstruction with Poisson Recon started at: ", time_start)
        mesh = me.poisson(pcd, depth)
        time_end = time.time()
        print("Mesh reconstruction with Poisson Recon ended at: ", time_end)
        print("Total Time Taken!", time_end - time_start)
        me.visualize_mesh(mesh)
        # me.write_mesh(mesh, result_mesh)
    
    


