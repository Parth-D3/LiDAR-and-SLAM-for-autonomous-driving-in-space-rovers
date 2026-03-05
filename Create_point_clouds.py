import numpy as np
import tifffile as tiff
import open3d as o3d
import os

dataset_path = "path-here"
folders = {
    "azimuth": os.path.join(dataset_path, "img_azimuth"),
    "elevation": os.path.join(dataset_path, "img_elevation"),
    "range": os.path.join(dataset_path, "img_range"),
    "intensity": os.path.join(dataset_path, "img_intensity16"),
    "time": os.path.join(dataset_path, "img_time"),
    "mask": os.path.join(dataset_path, "img_mask")
}

output_folder = os.path.join(dataset_path, "pointcloud_3d")
os.makedirs(output_folder, exist_ok=True)

frame_files = sorted(os.listdir(folders["azimuth"]))
frame_ids = [f.split("_img_azimuth.tif")[0] for f in frame_files]

for frame_id in frame_ids:
    az = tiff.imread(os.path.join(folders["azimuth"], frame_id + "_img_azimuth.tif"))
    el = tiff.imread(os.path.join(folders["elevation"], frame_id + "_img_elevation.tif"))
    r  = tiff.imread(os.path.join(folders["range"], frame_id + "_img_range.tif"))
    I  = tiff.imread(os.path.join(folders["intensity"], frame_id + "_img_intensity16.tif"))
    t  = tiff.imread(os.path.join(folders["time"], frame_id + "_img_time.tif"))
    mask = tiff.imread(os.path.join(folders["mask"], frame_id + "_img_mask.tif"))
    
    valid = (mask > 0) & (~np.isnan(r)) & (r > 0.1) 
    az, el, r, I, t = az[valid], el[valid], r[valid], I[valid], t[valid]
    
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    points = np.stack([x, y, z], axis=1)
    
    I = I.astype(np.float32)
    I_norm = (I - I.min()) / (I.max() - I.min() + 1e-6)
    colors = np.stack([I_norm, I_norm, I_norm], axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    pcd = pcd.voxel_down_sample(voxel_size=0.03)
    
    output_file = os.path.join(output_folder, frame_id + ".ply")
    o3d.io.write_point_cloud(output_file, pcd)
    
    print(f"Saved point cloud for frame {frame_id} -> {output_file}")
