#!/usr/bin/env python3
"""
Trajectory Evaluation against Ground Truth
============================================
Computes ATE (Absolute Trajectory Error) and RPE (Relative Pose Error)
between the estimated SLAM trajectory and MoCap ground truth.

Usage:
    python evaluate.py
    python evaluate.py --est slam_output/estimated_trajectory.txt --gt robot_data/extracted_data/mocap.txt
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import yaml


# ---------------------------------------------------------------------------
# TUM trajectory I/O
# ---------------------------------------------------------------------------

def load_tum_trajectory(path):
    """Load TUM-format trajectory → (timestamps, positions Nx3, quaternions Nx4)."""
    data = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            ts = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = (float(parts[4]), float(parts[5]),
                                float(parts[6]), float(parts[7]))
            data.append([ts, tx, ty, tz, qx, qy, qz, qw])

    data = np.array(data)
    return data[:, 0], data[:, 1:4], data[:, 4:8]


def associate_trajectories(ts_est, ts_gt, max_diff=0.05):
    """Associate estimated and ground truth poses by timestamp."""
    matches = []
    for i, t_e in enumerate(ts_est):
        diffs = np.abs(ts_gt - t_e)
        j = np.argmin(diffs)
        if diffs[j] <= max_diff:
            matches.append((i, j))
    return matches


# ---------------------------------------------------------------------------
# Umeyama alignment (SE3)
# ---------------------------------------------------------------------------

def umeyama_alignment(src, dst):
    """
    Umeyama alignment: find s, R, t such that dst ≈ s*R@src + t
    Returns: scale, rotation (3×3), translation (3,)
    """
    assert src.shape == dst.shape
    n, dim = src.shape

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    sigma_src = np.sum(src_c ** 2) / n
    cov = dst_c.T @ src_c / n

    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[dim - 1, dim - 1] = -1

    R = U @ S @ Vt
    scale = np.trace(np.diag(D) @ S) / sigma_src
    t = mu_dst - scale * R @ mu_src

    return scale, R, t


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def compute_ate(pos_est, pos_gt, align=True):
    """Absolute Trajectory Error (RMSE after optional Umeyama alignment)."""
    if align:
        scale, R, t = umeyama_alignment(pos_est, pos_gt)
        pos_aligned = (scale * (R @ pos_est.T).T) + t
    else:
        pos_aligned = pos_est

    errors = np.linalg.norm(pos_aligned - pos_gt, axis=1)
    rmse = np.sqrt(np.mean(errors ** 2))
    return rmse, errors, pos_aligned


def compute_rpe(pos_est, pos_gt, delta=1):
    """Relative Pose Error (translation, frame-to-frame)."""
    errors = []
    for i in range(len(pos_est) - delta):
        # Relative motion in estimated trajectory
        d_est = pos_est[i + delta] - pos_est[i]
        # Relative motion in ground truth
        d_gt = pos_gt[i + delta] - pos_gt[i]
        errors.append(np.linalg.norm(d_est - d_gt))

    errors = np.array(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    return rmse, errors


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_trajectories(pos_est, pos_gt, pos_aligned, ate_errors, rpe_errors,
                       output_dir):
    """Generate evaluation plots."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SLAM Trajectory Evaluation", fontsize=14, fontweight="bold")

    # 1. 2D trajectory (top-down: X-Y)
    ax = axes[0, 0]
    ax.plot(pos_gt[:, 0], pos_gt[:, 1], "g-", linewidth=2, label="Ground Truth")
    ax.plot(pos_aligned[:, 0], pos_aligned[:, 1], "r-", linewidth=1.5,
            label="Estimated (aligned)", alpha=0.8)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Trajectory (top-down)")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # 2. 3D trajectory
    ax = axes[0, 1]
    ax.remove()
    ax = fig.add_subplot(2, 2, 2, projection="3d")
    ax.plot3D(pos_gt[:, 0], pos_gt[:, 1], pos_gt[:, 2], "g-", linewidth=2,
              label="Ground Truth")
    ax.plot3D(pos_aligned[:, 0], pos_aligned[:, 1], pos_aligned[:, 2], "r-",
              linewidth=1.5, label="Estimated", alpha=0.8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Trajectory (3D)")
    ax.legend()

    # 3. ATE over time
    ax = axes[1, 0]
    ax.plot(ate_errors, "b-", linewidth=1)
    ax.axhline(np.mean(ate_errors), color="r", linestyle="--",
               label=f"Mean = {np.mean(ate_errors):.4f}m")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("ATE (m)")
    ax.set_title(f"Absolute Trajectory Error (RMSE={np.sqrt(np.mean(ate_errors**2)):.4f}m)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. RPE over time
    ax = axes[1, 1]
    ax.plot(rpe_errors, "m-", linewidth=1)
    ax.axhline(np.mean(rpe_errors), color="r", linestyle="--",
               label=f"Mean = {np.mean(rpe_errors):.4f}m")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("RPE (m)")
    ax.set_title(f"Relative Pose Error (RMSE={np.sqrt(np.mean(rpe_errors**2)):.4f}m)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "evaluation.png")
    plt.savefig(out_path, dpi=150)
    print(f"  ✓  Plot saved → {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate SLAM trajectory")
    parser.add_argument("--dataset", type=str, help="Name of the dataset (e.g., agv1_square_manual...) to automatically infer paths")
    parser.add_argument("--est", type=str, help="Path to estimated trajectory (overrides --dataset)")
    parser.add_argument("--gt", type=str, help="Path to ground truth trajectory (overrides --dataset)")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output", type=str, help="Output directory for plots (overrides --dataset)")
    parser.add_argument("--no-align", action="store_true",
                        help="Skip Umeyama alignment")
    args = parser.parse_args()

    if args.dataset:
        est_path = args.est or f"slam_output/{args.dataset}/orb_slam_results.txt"
        gt_path = args.gt or f"robot_data/extracted_data/{args.dataset}/mocap.txt"
        out_dir = args.output or f"slam_output/{args.dataset}/eval"
    else:
        est_path = args.est or "slam_output/estimated_trajectory.txt"
        gt_path = args.gt or "robot_data/extracted_data/mocap.txt"
        out_dir = args.output or "slam_output"

    print(f"\n{'='*60}")
    print(f"  Trajectory Evaluation")
    print(f"{'='*60}")

    if not os.path.isfile(est_path):
        print(f"  ✗  Estimated trajectory not found: {est_path}")
        sys.exit(1)
    if not os.path.isfile(gt_path):
        # Try fallback to gt_data directory
        if args.dataset:
            possible_gt = os.path.join("gt_data", args.dataset, "raw", "mocap.txt")
        else:
            possible_gt = "gt_data/mocap.txt"
            
        if os.path.isfile(possible_gt):
            gt_path = possible_gt
            print(f"  → Ground truth located in alternative path: {gt_path}")
        else:
            print(f"  ✗  Ground truth not found: {gt_path}")
            print("     (MoCap data may not have been recorded)")
            sys.exit(1)

    # Load trajectories
    ts_est, pos_est, quat_est = load_tum_trajectory(est_path)
    ts_gt, pos_gt, quat_gt = load_tum_trajectory(gt_path)

    # Filter out frozen poses at the end of the sequence where tracking was lost
    end_idx = len(pos_est)
    for i in range(len(pos_est) - 1, 20, -1):
        disp = np.linalg.norm(pos_est[i] - pos_est[i-20])
        if disp > 0.05:  # Found where the robot was actually moving (>5cm over 20 frames)
            end_idx = i + 1
            break
            
    ts_est = ts_est[:end_idx]
    pos_est = pos_est[:end_idx]
    quat_est = quat_est[:end_idx]
    
    if end_idx < len(pos_est) + 20: # just a sanity check
        print(f"  ⚠  Retained the first {end_idx} poses (cut off trailing tracking loss)\n")

    print(f"  Estimated : {len(ts_est)} poses")
    print(f"  Ground truth: {len(ts_gt)} poses")

    # Associate by timestamp
    matches = associate_trajectories(ts_est, ts_gt)
    print(f"  Matched   : {len(matches)} pose pairs\n")

    if len(matches) < 3:
        print("  ✗  Too few matches — check timestamp synchronisation.")
        sys.exit(1)

    idx_est = [m[0] for m in matches]
    idx_gt = [m[1] for m in matches]
    pos_est_m = pos_est[idx_est]
    pos_gt_m = pos_gt[idx_gt]

    # Convert Camera frame if this is ORB-SLAM data (Camera frame: Z-fwd, X-right, Y-down)
    # vs standard robotics frame for LiDAR/Odometry (X-fwd, Y-left, Z-up).
    if "orb" in est_path.lower():
        print("  → Detected ORB-SLAM trajectory: mapping Camera Optical Frame to MoCap frame")
        pos_est_m_converted = np.zeros_like(pos_est_m)
        pos_est_m_converted[:, 0] = pos_est_m[:, 2]   # X_gt = Z_est
        pos_est_m_converted[:, 1] = -pos_est_m[:, 0]  # Y_gt = -X_est
        pos_est_m_converted[:, 2] = -pos_est_m[:, 1]  # Z_gt = -Y_est
    else:
        pos_est_m_converted = pos_est_m

    # ATE
    align = not args.no_align
    ate_rmse, ate_errors, pos_aligned = compute_ate(pos_est_m_converted, pos_gt_m, align)
    print(f"  ATE RMSE : {ate_rmse:.4f} m")
    print(f"  ATE Mean : {np.mean(ate_errors):.4f} m")
    print(f"  ATE Max  : {np.max(ate_errors):.4f} m")

    # RPE
    rpe_rmse, rpe_errors = compute_rpe(pos_aligned, pos_gt_m)
    print(f"\n  RPE RMSE : {rpe_rmse:.4f} m")
    print(f"  RPE Mean : {np.mean(rpe_errors):.4f} m")

    print(f"\n{'='*60}\n")

    # Plot
    plot_trajectories(pos_est_m, pos_gt_m, pos_aligned, ate_errors, rpe_errors, out_dir)


if __name__ == "__main__":
    main()
