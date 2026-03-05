# LiDAR and SLAM for autonomous driving in space rovers

A computer vision based autonomous navigation system that uses LiDAR and SLAM for trajectory estimation and drift correction over long distances in planetary terrains for space rovers.

## Problem Statement

Autonomous navigation in unstructured outdoor environments is a fundamental challenge in robotics and space exploration. Rovers and autonomous vehicles operating in such environments need to accurately estimate their position and build a map of their surroundings without relying on external infrastructure such as GPS alone.

The core problem is **pose estimation** — determining where the vehicle is at every point in time as it moves through an environment. This is critical for path planning, obstacle avoidance, and autonomous decision making.

## Limitations of Traditional Approaches

Traditional methods for pose estimation rely on **feature extraction, feature matching, and visual odometry**.

### Feature Extraction and Matching
Algorithms such as SURF (Speeded-Up Robust Features) detect keypoints in sensor data and match them across consecutive frames to estimate relative motion. While effective over short distances, these methods suffer from a fundamental problem: **error accumulation**. Each frame-to-frame estimate carries a small error, and over hundreds or thousands of frames these errors compound, causing the estimated trajectory to drift significantly from the true path. In outdoor environments with repetitive terrain or sparse features, this drift becomes severe.

### Visual Odometry
Visual odometry estimates motion by tracking features across frames and computing the transformation between them. While it provides smooth local estimates, it has no mechanism to **correct past errors**. Once drift accumulates, there is no way to recover without an external reference. Over long distances, the estimated path can deviate by tens of meters from the true trajectory, making it unreliable for real world autonomous navigation.

### Depth Estimation
Stereo or monocular depth estimation methods struggle in environments with uniform textures, varying lighting, or dust, all of which are common in outdoor and planetary environments. Inaccurate depth estimates directly translate into inaccurate motion estimates, further compounding drift.

## Approach

To address these limitations, this project combines **LiDAR sensor data**, **SURF feature matching**, and **Graph SLAM** with GPS anchoring to produce accurate, drift-corrected trajectory estimates.

### LiDAR Data

The LiDAR sensor captures the environment by emitting laser pulses and measuring the time it takes for them to return. Each scan produces a set of points described in **spherical coordinates**:

- **Azimuth** — horizontal angle of the laser pulse
- **Elevation** — vertical angle of the laser pulse
- **Range** — distance to the reflecting surface
- **Intensity** — strength of the returned signal (16-bit)
- **Response** — quality measure of the detected feature

Each frame contains thousands of such points, collectively describing the 3D geometry of the surrounding environment.

### 3D Point Cloud Visualization

The spherical coordinates from the LiDAR scans are converted into **Cartesian 3D coordinates** using the following transformation:
```python
x = r * cos(elevation) * cos(azimuth)
y = r * cos(elevation) * sin(azimuth)
z = r * sin(elevation)
```

This produces a dense 3D point cloud for each frame. The 16-bit intensity values are normalized to 8-bit grayscale and assigned as colors to each point, producing an **intensity-colored 3D point cloud** that reflects the reflectivity of surfaces in the environment. These point clouds are saved in the PLY format and can be visualized in tools such as CloudCompare or Open3D, giving a clear 3D representation of the terrain the rover traversed.

## SLAM — Simultaneous Localization and Mapping

**SLAM** is the computational problem of constructing and updating a map of an unknown environment while simultaneously keeping track of the vehicle's location within it. Unlike visual odometry, SLAM maintains a global consistency by revisiting and correcting past pose estimates, effectively eliminating accumulated drift.

### Graph SLAM Implementation

This project implements **Graph SLAM** using the GTSAM (Georgia Tech Smoothing and Mapping) library. The core idea of Graph SLAM is to represent the trajectory as a **pose graph**:

- Each **node** in the graph represents the vehicle's pose (position and orientation) at a given frame
- Each **edge** represents the relative transformation between two consecutive poses, estimated from SURF feature matching

**Step 1 — Relative Motion Estimation**

For each consecutive pair of frames, SURF keypoints are extracted from the LiDAR data and matched across frames. A **rigid body transformation** (rotation matrix R and translation vector t) is estimated from the matched 3D point pairs using Singular Value Decomposition (SVD). This gives the relative motion between frames.

**Step 2 — Pose Graph Construction**

The relative transformations are added as **BetweenFactors** in the GTSAM factor graph. A **Huber robust loss function** is applied to reduce the influence of outlier matches, making the system robust to noisy or incorrect feature correspondences.

**Step 3 — GPS Anchoring**

To provide absolute position references and prevent global drift, **GPS measurements** are incorporated as **PriorFactors** in the graph. The GPS provides a strong positional constraint at each frame where a reading is available, anchoring the trajectory to real world coordinates while allowing SLAM to fill in the gaps between GPS readings with accurate local estimates.

**Step 4 — Pose Graph Optimization**

The full pose graph is optimized using the **Levenberg-Marquardt algorithm**, which minimizes the total error across all factors in the graph simultaneously. This global optimization corrects drift by adjusting all pose estimates together, ensuring the trajectory is globally consistent and locally accurate.

## Output

- Optimized SLAM trajectory plot
- GPS vs SLAM trajectory comparison

## Tech Stack

- Python
- GTSAM
- NumPy
- Pandas
- Matplotlib

## Installation

Clone the repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Update the file paths in `app.py` to point to your dataset, then run:
```bash
python3 app.py
```

## License

This project is licensed under CC BY-NC-ND 4.0. You may not use, modify, or distribute this work without explicit permission from the author.
