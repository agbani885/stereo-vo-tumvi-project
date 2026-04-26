# Stereo Visual Odometry on TUM VI

Classical geometry-based **Monocular → Metric Stereo Visual Odometry** (no deep learning).

**Master's Course Project in Computer Vision and Robotics**

## Pipeline Overview

1. **Data Loader + Calibration** (`scripts/01_data_loader.py`)
   - Loads TUM VI Room2 stereo images
   - Parses Kalibr calibration
   - Baseline B = **0.1011 meters**

2. **Monocular VO** (`scripts/03_monocular_vo.py`)
   - ORB features + Essential Matrix + RANSAC
   - Up-to-scale trajectory
   - ~2882 poses tracked

3. **Metric Stereo VO** (`scripts/04_stereo_vo.py`)
   - Extends monocular pipeline
   - Uses stereo baseline for metric scale
   - Outputs real-world metric trajectory
