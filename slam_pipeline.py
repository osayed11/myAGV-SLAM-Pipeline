#!/usr/bin/env python3
"""
RGB-D + LiDAR SLAM Pipeline (Open3D)
======================================
Offline multi-sensor SLAM using Open3D. Fuses RGB-D visual odometry with
2D LiDAR scan matching in a unified pose graph, producing a 3D
reconstruction and an optional 2D occupancy grid map.

Pipeline stages:
    1. Load RGB-D pairs + camera intrinsics + 2D LiDAR scans
    2. Frame-to-frame RGB-D odometry (optionally seeded by wheel odom)
    3. 2D LiDAR ICP scan matching (additional pose graph constraints)
    4. Keyframe-based pose graph construction (visual + LiDAR edges)
    5. Loop closure detection
    6. Global pose graph optimisation
    7. TSDF integration → dense point cloud & mesh
    8. 2D occupancy grid generation from LiDAR
    9. Save trajectory (TUM format), 3D model, and 2D map

Usage:
    python slam_pipeline.py                          # uses config.yaml
    python slam_pipeline.py --data robot_data/extracted_data/   # override data dir
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_intrinsics(data_dir, cfg):
    """Load camera intrinsics from extracted JSON or config defaults."""
    json_path = os.path.join(data_dir, "camera_intrinsics.json")
    if os.path.isfile(json_path):
        with open(json_path) as f:
            info = json.load(f)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            info["width"], info["height"],
            info["fx"], info["fy"], info["cx"], info["cy"]
        )
        print(f"  Intrinsics from {json_path}")
    else:
        cam = cfg["camera"]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            cam["width"], cam["height"],
            cam["fx"], cam["fy"], cam["cx"], cam["cy"]
        )
        print("  Intrinsics from config defaults")

    print(f"    {intrinsic.width}×{intrinsic.height}  "
          f"fx={intrinsic.intrinsic_matrix[0,0]:.1f}  "
          f"fy={intrinsic.intrinsic_matrix[1,1]:.1f}")
    return intrinsic


def load_associations(data_dir):
    """Load associations.txt → list of (rgb_path, depth_path, timestamp)."""
    assoc_path = os.path.join(data_dir, "associations.txt")
    if not os.path.isfile(assoc_path):
        print(f"  ✗  associations.txt not found in {data_dir}")
        print(f"     Run extract_bag.py first to generate it.")
        return []

    pairs = []
    with open(assoc_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            rgb_ts = float(parts[0])
            rgb_file = os.path.join(data_dir, parts[1])
            depth_file = os.path.join(data_dir, parts[3])
            # Verify files exist
            if not os.path.isfile(rgb_file):
                print(f"  ⚠  Missing RGB: {rgb_file} — skipping pair")
                continue
            if not os.path.isfile(depth_file):
                print(f"  ⚠  Missing depth: {depth_file} — skipping pair")
                continue
            pairs.append((rgb_file, depth_file, rgb_ts))
    return pairs


def load_wheel_odom(data_dir):
    """Load wheel odometry as dict {timestamp → 4×4 pose}."""
    odom_path = os.path.join(data_dir, "odom.txt")
    if not os.path.isfile(odom_path):
        print("  ⚠  odom.txt not found — wheel odom prior disabled.")
        return None

    from scipy.spatial.transform import Rotation
    poses = {}
    with open(odom_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            ts = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = (float(parts[4]), float(parts[5]),
                                float(parts[6]), float(parts[7]))
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]
            poses[ts] = T

    if not poses:
        print("  ⚠  odom.txt is empty — wheel odom prior disabled.")
        return None

    return poses


def make_rgbd(rgb_path, depth_path, depth_scale, depth_trunc):
    """Create an Open3D RGBDImage from file paths."""
    color = o3d.io.read_image(rgb_path)
    depth = o3d.io.read_image(depth_path)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth,
        depth_scale=1.0 / depth_scale,   # Open3D expects depth_scale = 1/factor
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False
    )
    return rgbd


def get_odom_prior(odom_dict, ts_src, ts_tgt):
    """Get relative transform between two timestamps from wheel odom."""
    if odom_dict is None:
        return np.eye(4)

    odom_times = np.array(sorted(odom_dict.keys()))

    def nearest_pose(ts):
        idx = np.argmin(np.abs(odom_times - ts))
        return odom_dict[odom_times[idx]]

    T_src = nearest_pose(ts_src)
    T_tgt = nearest_pose(ts_tgt)
    
    # Open3D expects target_to_source transform: T_src^{-1} @ T_tgt
    T_rel_ros = np.linalg.inv(T_src) @ T_tgt

    # Convert relative transform from ROS (X-forward) to Camera (Z-forward)
    # P_cam = T_ros_to_cam @ P_ros
    T_ros_to_cam = np.array([
        [ 0, -1,  0,  0],
        [ 0,  0, -1,  0],
        [ 1,  0,  0,  0],
        [ 0,  0,  0,  1]
    ])
    T_cam_to_ros = np.linalg.inv(T_ros_to_cam)
    
    T_rel_cam = T_ros_to_cam @ T_rel_ros @ T_cam_to_ros
    return T_rel_cam


# ---------------------------------------------------------------------------
# Odometry
# ---------------------------------------------------------------------------

def compute_pairwise_odometry(source_rgbd, target_rgbd, intrinsic, cfg,
                               init_transform=None):
    """Compute RGB-D odometry between two frames."""
    if init_transform is None:
        init_transform = np.eye(4)

    option = o3d.pipelines.odometry.OdometryOption()
    option.depth_diff_max = 0.5  # Increased from 0.07 to prevent tracking failure
    option.depth_min = cfg["slam"]["min_depth"]
    option.depth_max = cfg["slam"]["max_depth"]

    method_name = cfg["slam"].get("odometry_method", "hybrid")
    if method_name == "color":
        jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm()
    elif method_name == "point_to_plane":
        jacobian = (
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
        )
    else:  # hybrid (default)
        jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()

    success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
        source_rgbd, target_rgbd,
        intrinsic,
        init_transform,
        jacobian,
        option
    )
    return success, trans, info


# ---------------------------------------------------------------------------
# 2D LiDAR scan matching
# ---------------------------------------------------------------------------

def load_lidar_scans(data_dir):
    """Load extracted 2D LiDAR scans and metadata.

    Returns:
        scans: list of (timestamp, ranges_array)
        metadata: dict with angle_min, angle_max, etc.
    """
    lidar_dir = os.path.join(data_dir, "lidar")
    meta_path = os.path.join(data_dir, "scan_metadata.json")

    if not os.path.isdir(lidar_dir):
        print("  ⚠  lidar/ directory not found — LiDAR fusion disabled.")
        return [], None

    if not os.path.isfile(meta_path):
        print("  ⚠  scan_metadata.json not found — LiDAR fusion disabled.")
        return [], None

    with open(meta_path) as f:
        metadata = json.load(f)

    scan_files = sorted(
        [fn for fn in os.listdir(lidar_dir) if fn.endswith(".npy")]
    )

    if not scan_files:
        print("  ⚠  No .npy scan files found — LiDAR fusion disabled.")
        return [], None

    scans = []
    for fn in scan_files:
        ts = float(fn.replace(".npy", ""))
        ranges = np.load(os.path.join(lidar_dir, fn))
        scans.append((ts, ranges))

    print(f"  ✓  Loaded {len(scans)} LiDAR scans")
    return scans, metadata


def scan_to_pointcloud(ranges, metadata):
    """Convert a 2D laser scan (polar) to an Open3D PointCloud (x, y, z=0)."""
    angle_min = metadata["angle_min"]
    angle_inc = metadata["angle_increment"]
    r_min = metadata["range_min"]
    r_max = metadata["range_max"]

    angles = angle_min + np.arange(len(ranges)) * angle_inc

    # Filter invalid ranges
    valid = (ranges >= r_min) & (ranges <= r_max) & np.isfinite(ranges)
    r = ranges[valid]
    a = angles[valid]

    # Polar → Cartesian in Camera Optical Frame (Z-forward, X-right, Y-down)
    # Original ROS: x_ros = r*cos(a), y_ros = r*sin(a)
    # Mapping to Camera: Z_cam = x_ros, X_cam = -y_ros, Y_cam = 0
    z = r * np.cos(a)
    x = -r * np.sin(a)
    y = np.zeros_like(x)

    points = np.column_stack([x, y, z])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def match_scans_icp(source_pcd, target_pcd, max_distance, init_transform=None):
    """ICP registration between two 2D scan point clouds.

    Returns:
        success: bool
        transform: 4×4 relative transform
        information: 6×6 information matrix
        fitness: float (0-1)
    """
    if init_transform is None:
        init_transform = np.eye(4)

    result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd,
        max_distance,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=50
        )
    )

    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source_pcd, target_pcd, max_distance, result.transformation
    )

    return result.fitness > 0.1, result.transformation, info, result.fitness


def get_nearest_scan(scans, target_ts, max_diff=0.1):
    """Find the scan closest to target_ts within max_diff seconds."""
    best_idx = None
    best_diff = float("inf")
    for idx, (ts, _) in enumerate(scans):
        diff = abs(ts - target_ts)
        if diff < best_diff:
            best_diff = diff
            best_idx = idx
    if best_diff <= max_diff:
        return best_idx
    return None


# ---------------------------------------------------------------------------
# Pose graph
# ---------------------------------------------------------------------------

def build_pose_graph(pairs, intrinsic, cfg, odom_dict=None,
                     lidar_scans=None, lidar_meta=None):
    """
    Build a pose graph from RGB-D frame pairs + optional LiDAR scans.

    - RGB-D odometry edges between consecutive keyframes
    - 2D LiDAR ICP edges between consecutive keyframes (if available)
    - Loop closure edges between distant keyframes
    """
    slam_cfg = cfg["slam"]
    depth_scale = cfg["camera"]["depth_scale"]
    depth_trunc = slam_cfg["max_depth"]
    kf_interval = slam_cfg["keyframe_interval"]

    # Select keyframes
    kf_indices = list(range(0, len(pairs), kf_interval))
    n_kf = len(kf_indices)
    print(f"\n  Keyframes: {n_kf} (from {len(pairs)} total frames)")

    # Preload keyframe RGBD images
    print("  Loading keyframe images...")
    kf_rgbds = []
    kf_timestamps = []
    skipped = 0
    for ki in tqdm(kf_indices, desc="  Keyframes", unit="kf"):
        rgb_p, depth_p, ts = pairs[ki]
        try:
            rgbd = make_rgbd(rgb_p, depth_p, depth_scale, depth_trunc)
            kf_rgbds.append(rgbd)
            kf_timestamps.append(ts)
        except Exception as e:
            print(f"\n    ⚠  Failed to load keyframe {ki}: {e} — skipping")
            skipped += 1

    if skipped:
        print(f"  ⚠  Skipped {skipped} corrupted keyframes")
        # Rebuild kf_indices to match what was actually loaded
        kf_indices = [ki for idx, ki in enumerate(kf_indices)
                      if idx < len(kf_rgbds) + skipped][:len(kf_rgbds)]

    # Initialise pose graph
    pose_graph = o3d.pipelines.registration.PoseGraph()
    pose_graph.nodes.append(
        o3d.pipelines.registration.PoseGraphNode(np.eye(4))
    )

    # --- Odometry edges ---
    print("\n  Computing odometry edges...")
    global_pose = np.eye(4)
    for i in tqdm(range(n_kf - 1), desc="  Odometry", unit="edge"):
        init = get_odom_prior(odom_dict, kf_timestamps[i], kf_timestamps[i + 1])

        success, trans, info = compute_pairwise_odometry(
            kf_rgbds[i], kf_rgbds[i + 1], intrinsic, cfg, init
        )

        if not success:
            print(f"    ⚠  Odometry failed at keyframe {i}→{i+1}, using odom prior")
            trans = init
            info = np.eye(6) * 100  # low confidence

        global_pose = global_pose @ trans
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(global_pose))
        )
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                i, i + 1, trans, info, uncertain=False
            )
        )

    # --- LiDAR scan-matching edges ---
    lidar_cfg = slam_cfg.get("lidar", {})
    if lidar_cfg.get("enabled", False) and lidar_scans and lidar_meta:
        icp_max_dist = lidar_cfg.get("icp_max_distance", 0.5)
        fitness_thresh = lidar_cfg.get("icp_fitness_threshold", 0.3)
        lidar_weight = lidar_cfg.get("weight", 1.0)
        print(f"\n  Adding LiDAR scan-matching edges (ICP, max_dist={icp_max_dist}m)...")
        n_lidar_edges = 0

        for i in tqdm(range(n_kf - 1), desc="  LiDAR ICP", unit="edge"):
            # Find scans nearest to each keyframe timestamp
            idx_src = get_nearest_scan(lidar_scans, kf_timestamps[i])
            idx_tgt = get_nearest_scan(lidar_scans, kf_timestamps[i + 1])

            if idx_src is None or idx_tgt is None:
                continue

            src_pcd = scan_to_pointcloud(lidar_scans[idx_src][1], lidar_meta)
            tgt_pcd = scan_to_pointcloud(lidar_scans[idx_tgt][1], lidar_meta)

            if len(src_pcd.points) < 10 or len(tgt_pcd.points) < 10:
                continue

            # Use visual odometry transform as initial guess for ICP
            init_trans = pose_graph.edges[i].transformation

            success, trans, info, fitness = match_scans_icp(
                src_pcd, tgt_pcd, icp_max_dist, init_transform=init_trans
            )

            if success and fitness >= fitness_thresh:
                # Scale the information matrix by the configured weight
                info_weighted = info * lidar_weight
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        i, i + 1, trans, info_weighted, uncertain=False
                    )
                )
                n_lidar_edges += 1

        print(f"  ✓  Added {n_lidar_edges} LiDAR edges to pose graph")
    elif lidar_cfg.get("enabled", False):
        print("\n  ⚠  LiDAR fusion enabled but no scans available — skipping.")

    # --- Loop closure edges ---
    lc_cfg = slam_cfg.get("loop_closure", {})
    if lc_cfg.get("enabled", True):
        search_interval = lc_cfg.get("search_interval", 30)
        dist_thresh = lc_cfg.get("distance_threshold", 0.15)
        print(f"\n  Detecting loop closures (interval={search_interval})...")
        n_loops = 0

        for i in tqdm(range(0, n_kf, search_interval), desc="  Loop closure", unit="check"):
            for j in range(0, i - search_interval, search_interval):
                success, trans, info = compute_pairwise_odometry(
                    kf_rgbds[i], kf_rgbds[j], intrinsic, cfg
                )
                if success:
                    # Check transform magnitude (translation norm)
                    t_norm = np.linalg.norm(trans[:3, 3])
                    if t_norm < 5.0:  # sanity: not too far
                        pose_graph.edges.append(
                            o3d.pipelines.registration.PoseGraphEdge(
                                i, j, trans, info, uncertain=True
                            )
                        )
                        n_loops += 1

        print(f"  ✓  Found {n_loops} loop closures")

    return pose_graph, kf_indices, kf_timestamps


def optimise_pose_graph(pose_graph, cfg):
    """Modular global pose graph optimisation."""
    
    opt_type = cfg["slam"].get("optimizer", "lm").lower()
    
    if opt_type == "admm":
        print("\n  Optimising pose graph (Distributed Consensus ADMM)...")
        from admm_optimizer import optimise_pose_graph_admm
        pose_graph = optimise_pose_graph_admm(pose_graph, num_iterations=15, rho=1.0)
        print("  ✓  Pose graph optimised via ADMM")
        
    elif opt_type == "gbp":
        print("\n  Optimising pose graph (Gaussian Belief Propagation)...")
        from gbp_optimizer import optimise_pose_graph_gbp
        pose_graph = optimise_pose_graph_gbp(pose_graph, num_iterations=15)
        print("  ✓  Pose graph optimised via GBP")
        
    else:
        # Default Open3D Levenberg-Marquardt (Centralized)
        print("\n  Optimising pose graph (Open3D Levenberg-Marquardt)...")
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=0.05,
            edge_prune_threshold=0.25,
            preference_loop_closure=2.0,
            reference_node=0,
        )
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option,
        )
        print("  ✓  Pose graph optimised via LM")

    return pose_graph


# ---------------------------------------------------------------------------
# TSDF integration
# ---------------------------------------------------------------------------

def integrate_tsdf(pairs, kf_indices, pose_graph, intrinsic, cfg):
    """Integrate keyframe RGB-D images into a TSDF volume."""
    slam_cfg = cfg["slam"]
    depth_scale = cfg["camera"]["depth_scale"]
    voxel_size = slam_cfg["voxel_size"]
    sdf_trunc = slam_cfg["sdf_trunc"]
    depth_trunc = slam_cfg["max_depth"]

    print(f"\n  TSDF integration (voxel={voxel_size}m, trunc={sdf_trunc}m)...")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for node_idx, ki in enumerate(tqdm(kf_indices, desc="  Integrating", unit="kf")):
        if node_idx >= len(pose_graph.nodes):
            break
        rgb_p, depth_p, _ = pairs[ki]
        rgbd = make_rgbd(rgb_p, depth_p, depth_scale, depth_trunc)
        pose = pose_graph.nodes[node_idx].pose
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))

    print("  Extracting mesh and point cloud...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    pcd = volume.extract_point_cloud()

    print(f"  ✓  Mesh: {len(mesh.vertices)} vertices, "
          f"{len(mesh.triangles)} triangles")
    print(f"  ✓  Point cloud: {len(pcd.points)} points")

    return mesh, pcd


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_trajectory(pose_graph, kf_timestamps, output_path):
    """Save estimated trajectory in TUM format."""
    from scipy.spatial.transform import Rotation

    lines = []
    for i, ts in enumerate(kf_timestamps):
        if i >= len(pose_graph.nodes):
            break
        pose = pose_graph.nodes[i].pose
        t = pose[:3, 3]
        q = Rotation.from_matrix(pose[:3, :3]).as_quat()  # [qx,qy,qz,qw]
        lines.append(
            f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
            f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"
        )

    with open(output_path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        f.write("\n".join(lines) + "\n")

    print(f"  ✓  Trajectory ({len(lines)} poses) → {output_path}")


def save_results(mesh, pcd, pose_graph, kf_timestamps, output_dir):
    """Save all outputs."""
    os.makedirs(output_dir, exist_ok=True)

    # Trajectory
    save_trajectory(
        pose_graph, kf_timestamps,
        os.path.join(output_dir, "estimated_trajectory.txt")
    )

    # 3D models
    mesh_path = os.path.join(output_dir, "reconstruction.ply")
    pcd_path = os.path.join(output_dir, "pointcloud.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    o3d.io.write_point_cloud(pcd_path, pcd)
    print(f"  ✓  Mesh          → {mesh_path}")
    print(f"  ✓  Point cloud   → {pcd_path}")


# ---------------------------------------------------------------------------
# 2D Occupancy grid
# ---------------------------------------------------------------------------

def generate_occupancy_grid(lidar_scans, lidar_meta, pose_graph,
                            kf_timestamps, cfg, output_dir):
    """Generate a 2D occupancy grid map from LiDAR scans using optimised poses.

    Uses log-odds Bresenham ray-casting to mark free and occupied cells.
    """
    grid_cfg = cfg["slam"].get("occupancy_grid", {})
    if not grid_cfg.get("enabled", False):
        return
    if not lidar_scans or not lidar_meta:
        print("  ⚠  No LiDAR data — skipping occupancy grid.")
        return

    resolution = grid_cfg.get("resolution", 0.05)  # m/pixel
    map_size = grid_cfg.get("map_size", 30.0)       # metres
    occ_thresh = grid_cfg.get("occupied_threshold", 0.7)
    free_thresh = grid_cfg.get("free_threshold", 0.3)

    grid_dim = int(map_size / resolution)
    origin = grid_dim // 2  # robot starts at centre

    # Log-odds grid (0 = unknown)
    log_odds = np.zeros((grid_dim, grid_dim), dtype=np.float32)
    l_occ = np.log(0.9 / 0.1)   # log-odds for occupied
    l_free = np.log(0.3 / 0.7)  # log-odds for free

    print(f"\n  Generating 2D occupancy grid ({grid_dim}×{grid_dim}, "
          f"res={resolution}m)...")

    angle_min = lidar_meta["angle_min"]
    angle_inc = lidar_meta["angle_increment"]
    r_min = lidar_meta["range_min"]
    r_max = lidar_meta["range_max"]

    n_integrated = 0
    for node_idx, ts in enumerate(kf_timestamps):
        if node_idx >= len(pose_graph.nodes):
            break

        scan_idx = get_nearest_scan(lidar_scans, ts)
        if scan_idx is None:
            continue

        _, ranges = lidar_scans[scan_idx]
        pose = pose_graph.nodes[node_idx].pose

        # Robot position in grid coords
        # Camera frame: ground plane is X and Z (Y is down/height)
        rx = pose[0, 3] / resolution + origin
        ry = pose[2, 3] / resolution + origin
        # Yaw is rotation around the camera's Y-axis
        yaw = np.arctan2(pose[0, 2], pose[0, 0])

        angles = angle_min + np.arange(len(ranges)) * angle_inc

        for j in range(len(ranges)):
            r = ranges[j]
            if not np.isfinite(r) or r < r_min or r > r_max:
                continue

            # Endpoint in world frame
            beam_angle = yaw + angles[j]
            ex = rx + (r / resolution) * np.cos(beam_angle)
            ey = ry + (r / resolution) * np.sin(beam_angle)

            # Bresenham ray trace: mark cells along the ray as free
            x0, y0 = int(round(rx)), int(round(ry))
            x1, y1 = int(round(ex)), int(round(ey))
            for bx, by in _bresenham(x0, y0, x1, y1):
                if 0 <= bx < grid_dim and 0 <= by < grid_dim:
                    log_odds[by, bx] += l_free

            # Mark endpoint as occupied
            if 0 <= x1 < grid_dim and 0 <= y1 < grid_dim:
                log_odds[y1, x1] += l_occ

        n_integrated += 1

    # Clamp log-odds
    log_odds = np.clip(log_odds, -10, 10)

    # Convert to probability
    prob = 1.0 - 1.0 / (1.0 + np.exp(log_odds))

    # Create grayscale image: 0=occupied (black), 255=free (white), 128=unknown
    grid_img = np.full((grid_dim, grid_dim), 128, dtype=np.uint8)
    grid_img[prob > occ_thresh] = 0     # occupied
    grid_img[prob < free_thresh] = 255  # free

    os.makedirs(output_dir, exist_ok=True)
    map_path = os.path.join(output_dir, "occupancy_grid.png")
    cv2.imwrite(map_path, grid_img)

    # Save map metadata (for navigation use)
    map_meta = {
        "resolution": resolution,
        "origin_x": -map_size / 2,
        "origin_y": -map_size / 2,
        "width": grid_dim,
        "height": grid_dim,
        "occupied_threshold": occ_thresh,
        "free_threshold": free_thresh,
        "scans_integrated": n_integrated,
    }
    with open(os.path.join(output_dir, "occupancy_grid_meta.json"), "w") as f:
        json.dump(map_meta, f, indent=2)

    print(f"  ✓  Occupancy grid ({n_integrated} scans) → {map_path}")
    return grid_img


def _bresenham(x0, y0, x1, y1):
    """Bresenham's line algorithm — yields (x, y) cells along the ray."""
    cells = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return cells


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualise_results(mesh, pcd, pose_graph):
    """Interactive 3D viewer for the reconstruction."""
    print("\n  Launching 3D viewer (close window to continue)...")

    # Build trajectory line set
    traj_points = []
    for node in pose_graph.nodes:
        traj_points.append(node.pose[:3, 3])

    if len(traj_points) > 1:
        traj_lines = [[i, i + 1] for i in range(len(traj_points) - 1)]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(traj_points),
            lines=o3d.utility.Vector2iVector(traj_lines),
        )
        line_set.paint_uniform_color([1, 0, 0])  # red trajectory
        o3d.visualization.draw_geometries(
            [pcd, line_set],
            window_name="SLAM Reconstruction",
            width=1280, height=720,
        )
    else:
        o3d.visualization.draw_geometries(
            [pcd],
            window_name="SLAM Reconstruction",
            width=1280, height=720,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RGB-D SLAM Pipeline")
    parser.add_argument("--data", type=str, help="Extracted data directory")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output", type=str, default="slam_output")
    parser.add_argument("--no-vis", action="store_true", help="Skip visualisation")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = args.data or cfg["extraction"]["output_dir"]
    dataset_name = os.path.basename(os.path.normpath(data_dir))

    if args.output == "slam_output" and dataset_name != "extracted_data":
        output_dir = os.path.join(args.output, dataset_name)
    else:
        output_dir = args.output

    print(f"\n{'='*60}")
    print(f"  RGB-D + LiDAR SLAM Pipeline (Open3D)")
    print(f"{'='*60}")
    print(f"  Data   : {data_dir}")
    print(f"  Output : {output_dir}")
    print(f"{'='*60}\n")

    t0 = time.time()

    # 1. Load data
    print("[1/6] Loading data...")
    intrinsic = load_intrinsics(data_dir, cfg)
    pairs = load_associations(data_dir)
    print(f"  Loaded {len(pairs)} RGB-D pairs")

    if len(pairs) < 2:
        print("✗  Need at least 2 frames. Check extraction output.")
        sys.exit(1)

    odom_dict = None
    if cfg["slam"].get("use_wheel_odom_prior", False):
        print("  Loading wheel odometry prior...")
        odom_dict = load_wheel_odom(data_dir)
        if odom_dict:
            print(f"  ✓  {len(odom_dict)} odom poses loaded")
        else:
            print("  ℹ  Proceeding without wheel odom — using identity init for odometry.")

    # Load LiDAR scans
    lidar_scans, lidar_meta = [], None
    lidar_cfg = cfg["slam"].get("lidar", {})
    if lidar_cfg.get("enabled", False):
        print("  Loading LiDAR scans...")
        lidar_scans, lidar_meta = load_lidar_scans(data_dir)

    # 2-3. Build pose graph (visual odometry + LiDAR ICP + loop closures)
    print("\n[2/6] Building pose graph (RGB-D + LiDAR)...")
    pose_graph, kf_indices, kf_timestamps = build_pose_graph(
        pairs, intrinsic, cfg, odom_dict,
        lidar_scans=lidar_scans, lidar_meta=lidar_meta
    )

    # 4. Optimise
    print("\n[3/6] Pose graph optimisation...")
    pose_graph = optimise_pose_graph(pose_graph, cfg)

    # 5. TSDF integration
    print("\n[4/6] Dense reconstruction...")
    mesh, pcd = integrate_tsdf(pairs, kf_indices, pose_graph, intrinsic, cfg)

    # 6. 2D occupancy grid from LiDAR
    print("\n[5/6] 2D occupancy grid...")
    generate_occupancy_grid(
        lidar_scans, lidar_meta, pose_graph,
        kf_timestamps, cfg, output_dir
    )

    # 7. Save
    print("\n[6/6] Saving results...")
    save_results(mesh, pcd, pose_graph, kf_timestamps, output_dir)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  ✓  SLAM complete in {elapsed:.1f}s")
    print(f"{'='*60}\n")

    # Optional visualisation
    if not args.no_vis:
        visualise_results(mesh, pcd, pose_graph)


if __name__ == "__main__":
    main()
