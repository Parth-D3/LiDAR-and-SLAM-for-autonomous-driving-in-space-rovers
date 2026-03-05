import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gtsam
from gtsam import Pose3, Point3, Rot3, noiseModel
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.ParserWarning)

surf_folder  = "path_here"
match_folder = "path_here"
gps_file     = "path_here"
frame_prefix = "path_here"
frame_indices = range(1, 6273)
response_threshold = 0.15

def load_surf_csv(filepath):
    df = pd.read_csv(filepath, index_col=False)
    az, el, r = df["azimuth"].to_numpy(), df["elevation"].to_numpy(), df["range"].to_numpy()
    size, response = df["size"].to_numpy(), df["response"].to_numpy()
    return az, el, r, size, response

def surf_to_xyz(az, el, r):
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return np.column_stack((x, y, z))

def load_matches(filepath):
    df = pd.read_csv(filepath, index_col=False)
    return df.to_numpy(dtype=int)

def estimate_rigid_transform(A, B):
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
    t = centroid_B - R @ centroid_A
    return R, t

gps_df = pd.read_csv(gps_file, index_col=False)
gps_dict = {row['id']: np.array([row['x'], row['y'], row['z']]) for _, row in gps_df.iterrows()}

graph = gtsam.NonlinearFactorGraph()
initial = gtsam.Values()
hubermodel = gtsam.noiseModel.Robust.Create(
    gtsam.noiseModel.mEstimator.Huber(1.0),
    gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1,0.1,0.1,0.1,0.1,0.1]))
)

skipped_missing_files = 0
skipped_few_matches = 0

for i in frame_indices[:-1]:
    surf_A_path = os.path.join(surf_folder, f"path_here{i:08d}_surf.csv")
    surf_B_path = os.path.join(surf_folder, f"path_here{i+1:08d}_surf.csv")
    match_path  = os.path.join(match_folder, f"path_here{i:08d}_{i+1:08d}_matches.csv")

    try:
        az_A, el_A, r_A, size_A, resp_A = load_surf_csv(surf_A_path)
        az_B, el_B, r_B, size_B, resp_B = load_surf_csv(surf_B_path)
        matches = load_matches(match_path)
    except FileNotFoundError:
        skipped_missing_files += 1
        continue

    pts_A = surf_to_xyz(az_A, el_A, r_A)
    pts_B = surf_to_xyz(az_B, el_B, r_B)

    matched_A = pts_A[matches[:,0]]
    matched_B = pts_B[matches[:,1]]

    matched_resp_A = resp_A[matches[:,0]]
    matched_resp_B = resp_B[matches[:,1]]

    mask = (matched_resp_A > response_threshold) | (matched_resp_B > response_threshold)
    filtered_A = matched_A[mask]
    filtered_B = matched_B[mask]

    if len(filtered_A) < 3:
        skipped_few_matches += 1
        continue

    R, t = estimate_rigid_transform(filtered_A, filtered_B)
    T_rel = Pose3(Rot3(R), Point3(t))

    if not initial.exists(i):
        initial.insert(i, Pose3())
    if not initial.exists(i+1):
        initial.insert(i+1, T_rel.compose(initial.atPose3(i)))

    graph.add(gtsam.BetweenFactorPose3(i, i+1, T_rel, hubermodel))

    if (i+1) in gps_dict:
        gps_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6,1e-6,1e-6, 0.5,0.5,0.5]))
        gps_pose = Pose3(Rot3(), Point3(gps_dict[i+1]))
        graph.add(gtsam.PriorFactorPose3(i+1, gps_pose, gps_noise))

print(f"Skipped due to missing files: {skipped_missing_files}")
print(f"Skipped due to few matches: {skipped_few_matches}")

filtered_graph = gtsam.NonlinearFactorGraph()
for i in range(graph.size()):
    factor = graph.at(i)
    keys = factor.keys()
    if all(initial.exists(k) for k in keys):
        filtered_graph.add(factor)
    else:
        print(f"Skipping factor with missing keys: {[int(k) for k in keys]}")

params = gtsam.LevenbergMarquardtParams()
params.setVerbosityLM("SUMMARY")
optimizer = gtsam.LevenbergMarquardtOptimizer(filtered_graph, initial, params)
result = optimizer.optimize()

poses = []
for i in frame_indices:
    if result.exists(i):
        t = result.atPose3(i).translation()
        if isinstance(t, np.ndarray):
            poses.append([t[0], t[1], t[2]])
        else:
            poses.append([t.x(), t.y(), t.z()])

poses = np.array(poses)
plt.figure(figsize=(12,12))
plt.plot(poses[:,0], poses[:,1], '-o', color='blue', label='SLAM Trajectory')

gps_positions = np.array([gps_dict[i] for i in frame_indices if i in gps_dict])
if len(gps_positions):
    plt.plot(gps_positions[:,0], gps_positions[:,1], '-k', linewidth=2, label='GPS')

plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("Visual+GPS SLAM Trajectory")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()
