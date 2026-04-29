# RGB-D and Visual SLAM Pipelines

Offline multi-sensor SLAM framework for evaluating ORB-SLAM and Open3D pipelines against PhaseSpace MoCap ground truth.

## Project Directory Hierarchy

```
SLAM/
├── config.yaml                 # Central pipeline configuration
├── extract_bag.py              # ROS1 .bag sensor data extractor
├── evaluate.py                 # Trajectory evaluation suite (ATE/RPE metrics)
├── run_orbslam.sh              # Execution wrapper for monocular ORB-SLAM2
├── slam_pipeline.py            # Open3D Multi-Sensor SLAM pipeline
│
├── gt_data/
│   └── <sequence_name>/raw/
│       ├── mocap.bag           # Separate PhaseSpace ground-truth recordings
│       └── mocap.txt           # Extracted TUM ground-truth poses
│
├── robot_data/
│   ├── <sequence_name>.bag     # Main robot data recordings
│   └── extracted_data/         # Extracted visuals, logs, and LiDAR sweeps
│
└── slam_output/
    └── <sequence_name>/        # Algorithmic estimates & output plots
```

## Setup

Ensure your Python 3.12 environment is properly resolved:

```bash
cd "SLAM"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### ORB-SLAM2 Setup

To run monocular tracking, fetch and compile the designated ORB workspace:

```bash
git clone https://github.com/UCL/COMP0222-249_25-26_ORB_SLAM2
cd COMP0222-249_25-26_ORB_SLAM2
./build.sh
```

---

## Pipeline Workflows

### 1. Data Extraction (`extract_bag.py`)
Extracts topics out of ROS logs. Configure defaults directly inside `config.yaml`.

```bash
python3 extract_bag.py [options]
```

* `--bag PATH` : Override target source `.bag`.
* `--config PATH` : Pass tailored variables (Default: `config.yaml`).
* `--output DIR` : Redirect generated assets appropriately.

### 2. Executing SLAM Pipelines

#### Option A: Multi-Sensor SLAM (`slam_pipeline.py`)
Fuses depth visuals, laser rangefinders, and dead-reckoning priors.

```bash
python3 slam_pipeline.py [options]
```

* `--data DIR` : Path mapping to isolated sensor extractions.
* `--config PATH` : Node setup mapping parameters.
* `--output DIR` : Save output summaries.
* `--no-vis` : Bypasses Open3D pointcloud GUIs automatically.

#### Option B: Monocular ORB-SLAM2
Processes optical flows directly.

```bash
./run_orbslam.sh run robot_data/extracted_data/<dataset_name>/camera.yaml robot_data/extracted_data/<dataset_name>/
```

### 3. Performance Evaluation (`evaluate.py`)

Compares visual odometry estimates against absolute PhaseSpace baselines.

```bash
python3 evaluate.py [options]
```

* `--dataset NAME` : Matches sequence tags automatically.
* `--est PATH` : Point directly to target comparisons.
* `--gt PATH` : Anchor benchmarks carefully.
* `--no-align` : Restrict Umeyama 3D scaling.
