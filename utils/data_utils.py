import os
import re
import numpy as np

def get_files(file_path):
    rgb_images = {}
    parent_folders = os.listdir(file_path)
    for parent_folder in parent_folders:
        children = os.listdir(f'{file_path}/{parent_folder}')
        for child in children:
            folder_images = []
            child_path = f'{file_path}/{parent_folder}/{child}'
            if(not os.path.isdir(child_path)):
                continue
            else:
                files = sorted(os.listdir(child_path))
                for file in files:
                    if("depth" in file):
                        continue
                    elif(file == "fragments"):
                        continue
                    else:
                        folder_images.append(f'{child_path}/{file}')
            rgb_images[f'{child}'] = sorted(folder_images, key=custom_sort_key)
    return rgb_images

def custom_sort_key(file_name):
    parts = re.split(r'(\d+)', file_name)
    return [int(part) if part.isdigit() else part for part in parts]

def get_pcd_path(root_path, scene_type):
    parent_folders = os.listdir(root_path)
    for parent in parent_folders:
        scenes = os.listdir(f'{root_path}/{parent}')
        for scene in scenes:
            if(scene == scene_type):
                scene_path = f'{root_path}/{parent}/{scene_type}/{scene_type}.pcd'
            else:
                continue
    return scene_path

def get_pose_path(root_path, scene_type):
    parent_folders = os.listdir(root_path)
    for parent in parent_folders:
        scenes = os.listdir(f'{root_path}/{parent}')
        for scene in scenes:
            if(scene == scene_type):
                pose_path = f'{root_path}/{parent}/{scene_type}/{scene_type}.pose'
                dest_path = f'{root_path}/{parent}/{scene_type}/{scene_type}_traj.log'
            else:
                continue
    return pose_path, dest_path

def get_paths(rgb_image_path):
    depth_image_path = rgb_image_path.split(".")[0] + "_depth." + rgb_image_path.split('.')[1]
    return rgb_image_path, depth_image_path

def quaternion_to_rotation_matrix(q):
    q = np.array(q)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([[1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w,
                   2 * x * z + 2 * y * w],
                  [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2,
                   2 * y * z - 2 * x * w],
                  [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w,
                   1 - 2 * x**2 - 2 * y**2]])
    return R

def generate_camera_trajectory(pose_file_path, destination_path):
    f = open(pose_file_path, "r")
    traj = open(destination_path, "w")
    lines = f.readlines()
    for line in lines:
        line = line.split(" ")
        line = [float(i) for i in line[2:]]
        rot = quaternion_to_rotation_matrix(line[:4])
        rot = np.vstack((rot, np.array([0, 0, 0])))
        rot = np.hstack((rot, np.array([[line[4]], [line[5]], [line[6]], [1]])))

        traj.write(f"0 0 0\n"
                   f"{rot[0, 0]} {rot[0, 1]} {rot[0, 2]} {rot[0, 3]}\n"
                   f"{rot[1, 0]} {rot[1, 1]} {rot[1, 2]} {rot[1, 3]}\n"
                   f"{rot[2, 0]} {rot[2, 1]} {rot[2, 2]} {rot[2, 3]}\n"
                   f"{rot[3, 0]} {rot[3, 1]} {rot[3, 2]} {rot[3, 3]}\n"
                   )
    return