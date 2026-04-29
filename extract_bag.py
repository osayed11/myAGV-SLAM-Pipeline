#!/usr/bin/env python3
"""
ROS 1 Bag Extractor for SLAM Pipeline
======================================
Extracts RGB, Depth, LiDAR, Odometry, IMU, and MoCap from a ROS 1 bag file
into an organised directory structure ready for offline SLAM processing.

Output structure:
    robot_data/extracted_data/
    ├── rgb/                 # RGB images as PNG
    ├── depth/               # 16-bit depth images as PNG
    ├── lidar/               # 2D laser scans as .npy
    ├── associations.txt     # timestamp-matched RGB-Depth pairs (TUM format)
    ├── odom.txt             # wheel odometry in TUM format
    ├── mocap.txt            # MoCap ground truth in TUM format
    ├── imu.csv              # IMU readings
    ├── camera_intrinsics.json
    └── scan_metadata.json

Usage:
    python extract_bag.py                         # uses config.yaml
    python extract_bag.py --bag /path/to/file.bag  # override bag path
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ns_to_sec(ns):
    """Convert nanoseconds to seconds (float)."""
    return ns / 1e9


def quat_to_list(orientation):
    """Extract quaternion as [qx, qy, qz, qw] from a ROS Quaternion."""
    return [orientation.x, orientation.y, orientation.z, orientation.w]


def pos_to_list(position):
    """Extract position as [tx, ty, tz] from a ROS Point/Vector3."""
    return [position.x, position.y, position.z]


def tum_line(timestamp_ns, position, orientation):
    """Format a single TUM trajectory line: ts tx ty tz qx qy qz qw"""
    ts = ns_to_sec(timestamp_ns)
    p = pos_to_list(position)
    q = quat_to_list(orientation)
    return f"{ts:.6f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"


def image_msg_to_numpy(msg):
    """Convert a sensor_msgs/Image message to a numpy array."""
    h, w = msg.height, msg.width
    encoding = msg.encoding

    data = bytes(msg.data) if not isinstance(msg.data, bytes) else msg.data

    if encoding in ("rgb8", "bgr8"):
        img = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
        if encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    elif encoding == "16UC1":
        return np.frombuffer(data, dtype=np.uint16).reshape(h, w)
    elif encoding == "32FC1":
        img_f = np.frombuffer(data, dtype=np.float32).reshape(h, w)
        # Convert to 16UC1 (millimetres) for consistency
        return (img_f * 1000.0).astype(np.uint16)
    elif encoding == "mono8":
        return np.frombuffer(data, dtype=np.uint8).reshape(h, w)
    elif encoding in ("8UC3",):
        return np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
    else:
        raise ValueError(f"Unsupported image encoding: {encoding}")


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------

def extract_images(reader, typestore, topic, output_dir, fmt, skip, desc):
    """Extract images from a topic, returning list of (timestamp_ns, filename)."""
    connections = [c for c in reader.connections if c.topic == topic]
    if not connections:
        print(f"  ⚠  Topic '{topic}' not found in bag — skipping.")
        return []

    os.makedirs(output_dir, exist_ok=True)
    results = []
    count = 0

    msgs = list(reader.messages(connections=connections))
    for _, timestamp, rawdata in tqdm(msgs, desc=desc, unit="img"):
        count += 1
        if count % skip != 0:
            continue
        msg = typestore.deserialize_ros1(rawdata, connections[0].msgtype)
        img = image_msg_to_numpy(msg)
        fname = f"{ns_to_sec(timestamp):.6f}.{fmt}"
        fpath = os.path.join(output_dir, fname)
        cv2.imwrite(fpath, img)
        results.append((timestamp, fname))

    print(f"  ✓  Extracted {len(results)} images from {topic}")
    return results


def extract_camera_info(reader, typestore, topic):
    """Extract camera intrinsics from the first CameraInfo message."""
    connections = [c for c in reader.connections if c.topic == topic]
    if not connections:
        print(f"  ⚠  Topic '{topic}' not found — using defaults from config.")
        return None

    for _, _, rawdata in reader.messages(connections=connections):
        msg = typestore.deserialize_ros1(rawdata, connections[0].msgtype)
        k = list(msg.K)  # 3x3 row-major
        d = list(msg.D)
        info = {
            "width": msg.width,
            "height": msg.height,
            "fx": k[0], "fy": k[4],
            "cx": k[2], "cy": k[5],
            "distortion_model": msg.distortion_model,
            "D": d,
        }
        return info
    return None


def extract_odom(reader, typestore, topic, output_path):
    """Extract odometry to TUM format file."""
    connections = [c for c in reader.connections if c.topic == topic]
    if not connections:
        print(f"  ⚠  Topic '{topic}' not found — skipping odometry.")
        return 0

    lines = []
    for _, timestamp, rawdata in reader.messages(connections=connections):
        msg = typestore.deserialize_ros1(rawdata, connections[0].msgtype)
        pose = msg.pose.pose
        lines.append(tum_line(timestamp, pose.position, pose.orientation))

    with open(output_path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        f.write("\n".join(lines) + "\n")

    print(f"  ✓  Extracted {len(lines)} odometry poses → {output_path}")
    return len(lines)


def extract_mocap(reader, typestore, topic, output_path):
    """Extract MoCap poses to TUM format.  Handles PoseStamped and Odometry."""
    connections = [c for c in reader.connections if c.topic == topic]
    if not connections:
        print(f"  ⚠  Topic '{topic}' not found — skipping MoCap ground truth.")
        return 0

    msgtype = connections[0].msgtype
    lines = []
    for _, timestamp, rawdata in reader.messages(connections=connections):
        msg = typestore.deserialize_ros1(rawdata, msgtype)
        # Handle both PoseStamped and Odometry message types
        if hasattr(msg, "pose") and hasattr(msg.pose, "pose"):
            pose = msg.pose.pose  # nav_msgs/Odometry
        elif hasattr(msg, "pose") and hasattr(msg.pose, "position"):
            pose = msg.pose  # geometry_msgs/PoseStamped
        else:
            continue
        lines.append(tum_line(timestamp, pose.position, pose.orientation))

    if lines:
        with open(output_path, "w") as f:
            f.write("# timestamp tx ty tz qx qy qz qw\n")
            f.write("\n".join(lines) + "\n")
        print(f"  ✓  Extracted {len(lines)} MoCap poses → {output_path}")
    return len(lines)


def extract_lidar(reader, typestore, topic, output_dir):
    """Extract 2D laser scans as numpy files + metadata."""
    connections = [c for c in reader.connections if c.topic == topic]
    if not connections:
        print(f"  ⚠  Topic '{topic}' not found — skipping LiDAR.")
        return 0

    os.makedirs(output_dir, exist_ok=True)
    metadata_saved = False
    count = 0

    msgs = list(reader.messages(connections=connections))
    for _, timestamp, rawdata in tqdm(msgs, desc="LiDAR scans", unit="scan"):
        msg = typestore.deserialize_ros1(rawdata, connections[0].msgtype)

        if not metadata_saved:
            meta = {
                "angle_min": float(msg.angle_min),
                "angle_max": float(msg.angle_max),
                "angle_increment": float(msg.angle_increment),
                "range_min": float(msg.range_min),
                "range_max": float(msg.range_max),
                "num_readings": len(msg.ranges),
            }
            with open(os.path.join(output_dir, "..", "scan_metadata.json"), "w") as f:
                json.dump(meta, f, indent=2)
            metadata_saved = True

        ts = ns_to_sec(timestamp)
        ranges = np.array(msg.ranges, dtype=np.float32)
        np.save(os.path.join(output_dir, f"{ts:.6f}.npy"), ranges)
        count += 1

    print(f"  ✓  Extracted {count} LiDAR scans → {output_dir}")
    return count


def extract_imu(reader, typestore, topic, output_path):
    """Extract IMU data to CSV. Handles full Imu, Accel-only, and Gyro-only msgs."""
    connections = [c for c in reader.connections if c.topic == topic]
    if not connections:
        print(f"  ⚠  Topic '{topic}' not found — skipping IMU.")
        return 0

    rows = []
    header = None
    for _, timestamp, rawdata in reader.messages(connections=connections):
        try:
            msg = typestore.deserialize_ros1(rawdata, connections[0].msgtype)
        except Exception as e:
            print(f"  ⚠  IMU deserialisation error: {e} — skipping message.")
            continue

        ts = ns_to_sec(timestamp)

        # Full sensor_msgs/Imu
        if hasattr(msg, 'linear_acceleration') and hasattr(msg, 'angular_velocity'):
            a = msg.linear_acceleration
            g = msg.angular_velocity
            # orientation may be all zeros if IMU doesn't provide it
            if hasattr(msg, 'orientation'):
                o = msg.orientation
                rows.append(f"{ts:.6f},{a.x:.6f},{a.y:.6f},{a.z:.6f},"
                             f"{g.x:.6f},{g.y:.6f},{g.z:.6f},"
                             f"{o.x:.6f},{o.y:.6f},{o.z:.6f},{o.w:.6f}")
                header = "timestamp,ax,ay,az,gx,gy,gz,qx,qy,qz,qw"
            else:
                rows.append(f"{ts:.6f},{a.x:.6f},{a.y:.6f},{a.z:.6f},"
                             f"{g.x:.6f},{g.y:.6f},{g.z:.6f}")
                header = "timestamp,ax,ay,az,gx,gy,gz"
        # Accel-only (sensor_msgs/Imu published on /camera/accel/sample)
        elif hasattr(msg, 'linear_acceleration'):
            a = msg.linear_acceleration
            rows.append(f"{ts:.6f},{a.x:.6f},{a.y:.6f},{a.z:.6f}")
            header = header or "timestamp,ax,ay,az"
        # Gyro-only
        elif hasattr(msg, 'angular_velocity'):
            g = msg.angular_velocity
            rows.append(f"{ts:.6f},{g.x:.6f},{g.y:.6f},{g.z:.6f}")
            header = header or "timestamp,gx,gy,gz"
        else:
            print(f"  ⚠  Unknown IMU message format on {topic} — skipping.")
            return 0

    if not rows:
        print(f"  ⚠  No valid IMU messages found on {topic}.")
        return 0

    with open(output_path, "w") as f:
        f.write((header or "timestamp,data") + "\n")
        f.write("\n".join(rows) + "\n")

    print(f"  ✓  Extracted {len(rows)} IMU samples → {output_path}")
    return len(rows)


def associate_rgb_depth(rgb_list, depth_list, max_diff_s):
    """Match RGB and depth frames by closest timestamp (within threshold)."""
    if not rgb_list or not depth_list:
        if not rgb_list:
            print("  ⚠  No RGB frames extracted — cannot create associations.")
        if not depth_list:
            print("  ⚠  No depth frames extracted — cannot create associations.")
        return []

    associations = []
    depth_ts = np.array([ns_to_sec(t) for t, _ in depth_list])

    for rgb_ns, rgb_fname in rgb_list:
        rgb_s = ns_to_sec(rgb_ns)
        idx = np.argmin(np.abs(depth_ts - rgb_s))
        diff = abs(depth_ts[idx] - rgb_s)
        if diff <= max_diff_s:
            depth_ns, depth_fname = depth_list[idx]
            associations.append((rgb_s, rgb_fname, depth_ts[idx], depth_fname))

    if not associations:
        print(f"  ⚠  No RGB-Depth pairs matched within {max_diff_s}s threshold.")

    return associations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract sensor data from a ROS 1 bag file")
    parser.add_argument("--bag", type=str, help="Path to .bag file (overrides config)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    parser.add_argument("--output", type=str, help="Output directory (overrides config)")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    bag_path = args.bag or cfg["bag_file"]
    bag_name = os.path.splitext(os.path.basename(bag_path))[0]
    base_output_dir = args.output or cfg["extraction"]["output_dir"]
    output_dir = os.path.join(base_output_dir, bag_name)
    topics = cfg["topics"]
    skip = cfg["extraction"].get("skip_frames", 1)
    fmt = cfg["extraction"].get("image_format", "png")
    max_diff = cfg["extraction"].get("max_time_diff_s", 0.05)

    if not os.path.isfile(bag_path):
        print(f"✗  Bag file not found: {bag_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  ROS Bag Extractor")
    print(f"{'='*60}")
    print(f"  Bag   : {bag_path}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore
    typestore = get_typestore(Stores.ROS1_NOETIC)

    with Reader(bag_path) as reader:
        # Print bag info
        duration = (reader.end_time - reader.start_time) / 1e9
        print(f"  Duration : {duration:.1f}s")
        print(f"  Topics   : {len(reader.connections)}")
        print(f"  Messages : {reader.message_count}\n")

        # 1. Camera intrinsics
        print("[1/7] Camera intrinsics...")
        intrinsics = extract_camera_info(reader, typestore, topics["camera_info"])
        if intrinsics:
            out_path = os.path.join(output_dir, "camera_intrinsics.json")
            os.makedirs(output_dir, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(intrinsics, f, indent=2)
            print(f"  ✓  Saved intrinsics → {out_path}")
        else:
            # Use defaults from config
            cam = cfg["camera"]
            intrinsics = {
                "width": cam["width"], "height": cam["height"],
                "fx": cam["fx"], "fy": cam["fy"],
                "cx": cam["cx"], "cy": cam["cy"],
                "D": [], "distortion_model": "none",
            }
            out_path = os.path.join(output_dir, "camera_intrinsics.json")
            os.makedirs(output_dir, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(intrinsics, f, indent=2)
            print(f"  ✓  Saved default intrinsics → {out_path}")

        # 2. RGB images
        print("\n[2/7] RGB images...")
        rgb_list = extract_images(
            reader, typestore, topics["rgb"],
            os.path.join(output_dir, "rgb"), fmt, skip, "RGB frames"
        )

        # 3. Depth images
        print("\n[3/7] Depth images...")
        depth_list = extract_images(
            reader, typestore, topics["depth"],
            os.path.join(output_dir, "depth"), "png", skip, "Depth frames"
        )

        # 4. Associations
        print("\n[4/7] Associating RGB-Depth pairs...")
        n_associations = 0
        associations = associate_rgb_depth(rgb_list, depth_list, max_diff)
        assoc_path = os.path.join(output_dir, "associations.txt")
        os.makedirs(output_dir, exist_ok=True)
        with open(assoc_path, "w") as f:
            f.write("# rgb_timestamp rgb_file depth_timestamp depth_file\n")
            for rgb_ts, rgb_f, d_ts, d_f in associations:
                f.write(f"{rgb_ts:.6f} rgb/{rgb_f} {d_ts:.6f} depth/{d_f}\n")
        n_associations = len(associations)
        print(f"  ✓  {n_associations} associated pairs → {assoc_path}")

        # 5. Odometry
        print("\n[5/7] Odometry...")
        n_odom = extract_odom(reader, typestore, topics["odom"],
                              os.path.join(output_dir, "odom.txt"))

        # 6. LiDAR
        print("\n[6/7] LiDAR scans...")
        n_lidar = extract_lidar(reader, typestore, topics["lidar"],
                                os.path.join(output_dir, "lidar"))

        # 7. MoCap ground truth
        mocap_bag = cfg.get("mocap_bag_file", "")
        if mocap_bag and os.path.isfile(mocap_bag):
            print(f"\n[7/7] MoCap ground truth (from separate bag)...")
            mocap_out_path = os.path.join(os.path.dirname(mocap_bag), "mocap.txt")
            with Reader(mocap_bag) as mocap_reader:
                mocap_reader.open()
                n_mocap = extract_mocap(mocap_reader, typestore, topics["mocap"], mocap_out_path)
        else:
            print("\n[7/7] MoCap ground truth...")
            n_mocap = extract_mocap(reader, typestore, topics["mocap"],
                                    os.path.join(output_dir, "mocap.txt"))

        # Optional: IMU
        print("\n[+] IMU data...")
        n_imu = extract_imu(reader, typestore, topics["imu"],
                            os.path.join(output_dir, "imu.csv"))

    # -----------------------------------------------------------------------
    # Summary report
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Extraction Summary")
    print(f"{'='*60}")

    summary = [
        ("Camera intrinsics", "✓" if intrinsics else "✗  MISSING (using defaults)"),
        ("RGB images",        f"✓  {len(rgb_list)}" if rgb_list else "✗  MISSING"),
        ("Depth images",      f"✓  {len(depth_list)}" if depth_list else "✗  MISSING"),
        ("RGB-D associations",f"✓  {n_associations}" if n_associations else "✗  NONE"),
        ("Wheel odometry",    f"✓  {n_odom}" if n_odom else "⚠  not recorded"),
        ("LiDAR scans",       f"✓  {n_lidar}" if n_lidar else "⚠  not recorded"),
        ("MoCap ground truth",f"✓  {n_mocap}" if n_mocap else "⚠  not recorded"),
        ("IMU",               f"✓  {n_imu}" if n_imu else "⚠  not recorded"),
    ]

    for name, status in summary:
        print(f"  {name:22s} {status}")

    # Critical warnings
    if not rgb_list or not depth_list or not n_associations:
        print(f"\n  ✗  CRITICAL: RGB-D data is incomplete — SLAM cannot run!")
        print(f"     Check that your bag contains the configured topics:")
        print(f"       RGB:   {topics['rgb']}")
        print(f"       Depth: {topics['depth']}")
    else:
        print(f"\n  ✓  Extraction complete!  →  {output_dir}/")
        if not n_odom:
            print(f"     Note: No odometry — SLAM will rely on visual odometry only.")
        if not n_mocap:
            print(f"     Note: No MoCap — evaluation against ground truth unavailable.")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
