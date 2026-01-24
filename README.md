# Satellite Localization: Visual Localization Using Pre-existing Satellite Images 
*Capstone Project | B.Sc. in Astronautical Engineering*

A robust visual localization pipeline designed as a complementary aiding source for inertial navigation systems in GNSS-denied environments. The system estimates aerial vehicle coordinates by matching onboard imagery against satellite map tiles, providing periodic absolute position fixes to correct INS drift. It utilizes aerial vehicle attitude angles to transform oblique footage into a precise nadir (top-down) view, enabling accurate cross-view matching.


<p align="center">
  <img src="trajectory_showcase.gif" width="850" alt="Consecutive Localization Showcase">
  <br>
  <i>GNSS-Free Coordinate Estimation: Ground Truth (Orange) vs. Estimated Position (Blue)</i>
</p>

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage (End-to-End Pipeline)](#usage-end-to-end-pipeline)
- [Methodology](#methodology)
- [Evaluation Scenarios](#evaluation-scenarios)
- [Future Enhancements](#future-enhancements)
- [References & Acknowledgments](#references--acknowledgments)

## Features

*   **Perspective Warping:** Automatically corrects aerial vehicle tilt (Pitch/Roll) and heading (Yaw) to align aerial vehicle imagery with satellite imagery.
*   **Deep Matchers:** Supports **GIM**, **LightGlue**, **LoFTR**, and **MINIMA**.
*   **Multi-Zoom Analysis:** Evaluate performance across different satellite resolutions (Zoom levels 15-18).
*   **Consecutive Localization:** Processes consecutive frames using **Displacement Prediction** (simulating INS/Odometry) to constrain the search window for real-time coordinate estimation.


## Installation

```bash
# Clone with submodules
git clone --recursive https://github.com/hamitbugrabayram/SatelliteLocalization.git
cd SatelliteLocalization

# Setup environment
conda create -n satloc python=3.9 -y && conda activate satloc
pip install -r requirements.txt

# Download Matcher Weights (Example for MINIMA)
cd matchers/MINIMA/weights && bash download.sh && cd ../../..
```

## Dataset Setup

The pipeline is tested with the **VisLoc** dataset. 

1.  **Extract Data**: Download and unzip `UAV_VisLoc_dataset.zip` into the root directory.
2.  **Structure**:
    ```
    _VisLoc_dataset/
    ├── 01/
    │   ├── 01.csv            # Lat, Lon, Alt, Omega, Phi1, Kappa
    │   └── drone/            # Oblique images (.JPG)
    └── satellite_coordinates_range.csv  # Region mappings
    ```

## Usage

The system uses `runner.py` to automate both benchmark scenarios and specialized localization tests.

### Scenario 1: Global Benchmark (Random Sampling)
Prepares and runs experiments on 20 random images per region across multiple zoom levels.
```bash
# Step 1: Prepare data and tiles
python runner.py --dataset-prepare --max-queries 20 --zoom-levels 16 17

# Step 2: Run localization
python runner.py --run --zoom-levels 16 17

# Step 3: Evaluate results
python runner.py --eval
```

### Scenario 2: Consecutive Localization Test
Processes a full flight path using displacement prediction and a tight local search radius.
```bash
python runner.py --consecutive-test [REGION_ID] --consecutive-zoom 16
```
*   **--consecutive-test**: Region ID (1-11).
*   **--consecutive-zoom**: Zoom level to use (Default: 17).
*   **--sample-interval**: Process every Nth frame (Default: 30).

## Methodology

### 1. Perspective Warping (Nadir Transformation)
Oblique aerial vehicle images are transformed into a top-down view using:
$$H_{warp} = K \cdot R_{AV}^T \cdot R_{nadir} \cdot K^{-1}$$

### 2. Displacement Prediction (Dead Reckoning)
In consecutive localization mode, the system predicts the next search center by adding the vehicle's displacement vector (calculated from consecutive GT/Sensor data) to the last successful match position. This significantly reduces the search space to a **1000m radius**.

### 3. Feature Matching & Geometric Verification
Dense correspondences are established using deep matchers. A Homography matrix is estimated via RANSAC with strict stability checks (Determinant, Area Purity, and Boundary Constraints).

## Evaluation Scenarios

### Scenario 1: Global Stability & Multi-Zoom Analysis
Testing global robustness by matching random samples against the entire map database.

| Region (ID) | Altitude | Zoom | Success Rate | Avg Inliers | Min Error | Max Error | Avg Error |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Changjiang_20 (01)** | 406m | 15 | 0.0% | 0.0 | N/A | N/A | N/A |
| | | 16 | 5.0% | 117.0 | 13.41m | 13.41m | 13.41m |
| | | 17 | 10.0% | 168.0 | 15.68m | 57.37m | 36.53m |
| | | 18 | 5.0% | 84.0 | 3.03m | 3.03m | 3.03m |
| **Changjiang_23 (02)** | 406m | 15 | 0.0% | 0.0 | N/A | N/A | N/A |
| | | 16 | 5.0% | 245.0 | 7.05m | 7.05m | 7.05m |
| | | 17 | 15.0% | 269.7 | 7.02m | 21.98m | 13.48m |
| | | 18 | 10.0% | 134.0 | 6.77m | 11.55m | 9.16m |
| **Taizhou_1 (03)** | 466m | 15 | 0.0% | 0.0 | N/A | N/A | N/A |
| | | 16 | 20.0% | 201.5 | 2.35m | 27.22m | 13.95m |
| | | 17 | 45.0% | 300.9 | 1.86m | 31.93m | 20.15m |
| | | 18 | 15.0% | 182.7 | 23.18m | 26.43m | 24.79m |
| **Taizhou_6 (04)** | 545m | 15 | 0.0% | 0.0 | N/A | N/A | N/A |
| | | 16 | 35.0% | 258.3 | 12.02m | 95.66m | 39.07m |
| | | 17 | 50.0% | 310.4 | 12.91m | 96.63m | 29.65m |
| | | 18 | 10.0% | 125.5 | 9.93m | 31.89m | 20.91m |
| **Yunnan (05)** | 2315m | 15 | 0.0% | 0.0 | N/A | N/A | N/A |
| | | 16 | 5.0% | 131.0 | 49.24m | 49.24m | 49.24m |
| | | 17 | 15.0% | 305.3 | 26.97m | 54.25m | 42.58m |
| | | 18 | 5.0% | 84.0 | 28.50m | 28.50m | 28.50m |
| **Zhuxi (06)** | 838m | 15 | 0.0% | 0.0 | N/A | N/A | N/A |
| | | 16 | 0.0% | 0.0 | N/A | N/A | N/A |
| | | 17 | 5.0% | 78.0 | 46.78m | 46.78m | 46.78m |
| | | 18 | 0.0% | 0.0 | N/A | N/A | N/A |
| **Donghuayuan (07)** | 690m | 15 | 0.0% | 0.0 | N/A | N/A | N/A |
| | | 16 | 0.0% | 0.0 | N/A | N/A | N/A |
| | | 17 | 10.0% | 114.5 | 0.55m | 2.95m | 1.75m |
| | | 18 | 15.0% | 231.0 | 2.02m | 3.29m | 2.66m |
| **Huzhou_3_P1 (08)** | 552m | 15 | 0.0% | 0.0 | N/A | N/A | N/A |
| | | 16 | 35.0% | 310.4 | 13.04m | 38.67m | 26.31m |
| | | 17 | 55.0% | 348.4 | 13.14m | 55.17m | 26.91m |
| | | 18 | 5.0% | 164.0 | 16.98m | 16.98m | 16.98m |
| **Huzhou_3_P2 (09)** | 547m | 15 | 0.0% | 0.0 | N/A | N/A | N/A |
| | | 16 | 25.0% | 494.6 | 3.16m | 32.77m | 19.20m |
| | | 17 | 45.0% | 305.7 | 5.39m | 50.27m | 25.15m |
| | | 18 | 5.0% | 125.0 | 3.72m | 3.72m | 3.72m |
| **Huailai (10)** | 772m | 15 | 0.0% | 0.0 | N/A | N/A | N/A |
| | | 16 | 0.0% | 0.0 | N/A | N/A | N/A |
| | | 17 | 20.0% | 146.0 | 19.34m | 26.83m | 23.76m |
| | | 18 | 5.0% | 196.0 | 26.50m | 26.50m | 26.50m |
| **Shandan (11)** | 2572m | 15 | 25.0% | 369.0 | 14.06m | 49.44m | 26.25m |
| | | 16 | 85.0% | 425.8 | 13.45m | 78.60m | 30.04m |
| | | 17 | 65.0% | 242.5 | 10.66m | 51.30m | 27.26m |
| | | 18 | 0.0% | 0.0 | N/A | N/A | N/A |

### Scenario 2: Consecutive Localization (Local Search & INS Simulation)
This scenario simulates a continuous flight path by processing the dataset consecutively at a **30-frame interval**. It demonstrates the system's ability to maintain a persistent state and leverage temporal consistency.

#### Key Technical Approaches
*   **Perspective Warping (Nadir Transformation):** Oblique aerial imagery is transformed into top-down view using gimbal angles (Pitch/Roll/Yaw) and camera intrinsics. The adaptive yaw feature automatically selects the nearest 90° multiple to minimize rotation distortion, preserving image quality for matching.
*   **INS/Odometry Simulation:** The system predicts the next search window by calculating the vehicle's displacement vector from consecutive telemetry data. This "dead reckoning" mimics an integrated Navigation System.
*   **Search Radius Optimization:** By predicting the search center, the search space is reduced from a global region to a tight **1000m local window**, dramatically improving matching speed and reducing false positives in repetitive terrains.
*   **Progressive Map Generation:** The system dynamically stitches satellite tiles and draws the paths live, saving a snapshot at every checkpoint for thorough analysis.

#### Visual Elements Legend
*   **Thick Orange Line:** The continuous Ground Truth path of the Aerial Vehicle.
*   **Mega Blue Dots:** Estimated positions successfully localized by the system.
*   **Thick Red Lines:** Visual representation of the precision gap (offset) between Estimated and actual coordinates at each checkpoint.
*   **Red 'X' Markers:** Localizations that failed geometric verification; the system uses displacement prediction to jump to the next window and re-acquire track.

### Future Enhancements
*   **Adaptive Zoom Selection:** Automatic zoom level based on real-time GSD.
*   **Map Database:** Efficient indexing for global retrieval.
*   **Recursive Search:** Expanding radius only upon consecutive failures.
*   **INS-Complementary Integration:** This visual localization system is not intended as a standalone navigation solution, but as a complementary aiding source for INS. By providing periodic absolute position fixes, it can correct INS drift during extended GNSS-denied operations, similar to how GNSS aids INS in traditional navigation systems.

## References & Acknowledgments
1.  **[WildNav (TIERS)](https://github.com/TIERS/wildnav):** Core cross-view concept inspiration.
2.  **[UAV-VisLoc Dataset](https://github.com/IntelliSensing/UAV-VisLoc):** Official benchmark data source.
3.  **[Aerial-Satellite-Imagery-Retrieval](https://github.com/chiragkhandhar/Aerial-Satellite-Imagery-Retrieval):** Modified Bing Maps retrieval system.
4.  **Deep Matchers:** 
    *   [GIM](https://github.com/xuelunshen/gim)
    *   [LightGlue](https://github.com/cvg/LightGlue)
    *   [LoFTR](https://github.com/zju3dv/LoFTR)
    *   [MINIMA](https://github.com/LSXI7/MINIMA)

---
*Developed as a B.Sc. Graduation Project.*
