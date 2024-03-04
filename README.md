## 3D Reconstruction Methods and Analysis:

### Dependencies:
1) Open3D
2) Python=3.8.*
3) Pytorch
4) Cuda=12.*

### Description:
Backend package to process RGBD images and reconstruct into meshes. The package follows three paths to generate a 3D Mesh:
1) Registered Pointcloud -> Mesh: Assumes that the scene pointcloud is available

2) RGBD -> Pointcloud -> Registration -> Combined Pointcloud -> Mesh: Assumes nothing. Performs registration on multiple pointclouds and merges 
the optimized pointclouds. It then performs mesh reconstruction. Though, ideal; it often doesn't converge well for some datasets. Also, 
extremely time consuming. Tested for Desk_1 dataset

3) RGBD -> Assumed trajectory -> Combined Pointcloud-> mesh: This assumes camera trajectory as a prior, registers the rgbd images based on known trajectory, creates a pointcloud and constructs a mesh. Works the best

config/config.py contains configurations which can be modified to test mesh reconstruction

### Run:
Args:
- scene_type: "desk_1", "background_1" etc,
- data_type: "pointcloud", "RGBD", "registration"
- recon_method: "poisson"

```
# To run the script, eg:
python src/main.py --data_type "RGBD" --scene_type "desk_1" --recon_method "poisson"
```

### To do:

- Add time measure for reconstruction comparison
- Add evaluation metrics
- UV Texture mapping