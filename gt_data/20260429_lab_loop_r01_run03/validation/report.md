# Dataset Run Validation

- Bag: `/home/adria/Distributed_SLAM/runs/20260429_lab_loop_r01_run03/raw/mocap.bag`
- Result: **FAIL**

## Required Topics

- [ ] `/tf`
- [x] `/tf_static`
- [x] `/phasespace/markers`
- [ ] `/gt/robot_01/pose`
- [ ] `/gt/robot_01/odom`

## Missing Topics

- `/tf`
- `/gt/robot_01/pose`
- `/gt/robot_01/odom`

## Topic Summary

| Topic | Type | Count | Rate Hz |
| --- | --- | ---: | ---: |
| `/gt/robot_01/status` | `std_msgs/String` | 75 | 0.999 |
| `/mocap_debug/status` | `std_msgs/String` | 77 | 1.000 |
| `/phasespace/cameras` | `phasespace_msgs/Cameras` | 46 | 0.973 |
| `/phasespace/markers` | `phasespace_msgs/Markers` | 70899 | 1009.946 |
| `/tf_static` | `tf2_msgs/TFMessage` | 2611 | 28244.471 |
