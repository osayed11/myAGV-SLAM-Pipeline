# Dataset Run Validation

- Bag: `/home/adria/Distributed_SLAM/runs/20260429_lab_loop_r01_run02/raw/mocap.bag`
- Result: **FAIL**

## Required Topics

- [x] `/tf`
- [x] `/tf_static`
- [x] `/phasespace/markers`
- [x] `/gt/robot_01/pose`
- [x] `/gt/robot_01/odom`

## Non-Monotonic Timestamps

- `/gt/robot_01/pose`: 3 events
- `/tf`: 4 events
- `/gt/robot_01/odom`: 2 events
- `/gt/robot_01/path`: 2 events
- `/phasespace/markers`: 4 events

## Topic Summary

| Topic | Type | Count | Rate Hz |
| --- | --- | ---: | ---: |
| `/gt/robot_01/odom` | `nav_msgs/Odometry` | 40860 | 10782.272 |
| `/gt/robot_01/path` | `nav_msgs/Path` | 4086 | 46.869 |
| `/gt/robot_01/pose` | `geometry_msgs/PoseStamped` | 40860 | 764.408 |
| `/gt/robot_01/status` | `std_msgs/String` | 60 | 0.681 |
| `/mocap_debug/status` | `std_msgs/String` | 90 | 1.000 |
| `/phasespace/cameras` | `phasespace_msgs/Cameras` | 52 | 0.964 |
| `/phasespace/markers` | `phasespace_msgs/Markers` | 78371 | 2551.280 |
| `/tf` | `tf2_msgs/TFMessage` | 40859 | 755.186 |
| `/tf_static` | `tf2_msgs/TFMessage` | 2959 | 28339.892 |
