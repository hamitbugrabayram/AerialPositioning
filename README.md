# Aerial Positioning: Visual Positioning for Aerial Imagery Using Pre-existing Satellite Images 
*Capstone Project | B.Sc. in Astronautical Engineering*

A robust vision-based positioning pipeline designed as a complementary aiding source for inertial navigation systems (INS) in GNSS-denied environments. The system estimates the horizontal position (latitude and longitude) of an aerial vehicle by matching onboard camera imagery with publicly available satellite map tiles. The resulting position estimates are used to periodically correct INS drift, preventing the accumulation of long-term positioning errors. To enable reliable cross-view matching, the pipeline leverages vehicle attitude information to warp oblique camera views into a nadir (top-down) perspective aligned with satellite imagery.

<p align="center">
  <img src="positioning_showcase.gif" width="850" alt="Visual Positioning Showcase">
  <br>
  <i>GNSS-Free Coordinate Estimation: Ground Truth (Orange) vs. Estimated Position (Blue)</i>
</p>

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Methodology](#methodology)
- [Future Enhancements](#future-enhancements)
- [References & Acknowledgments](#references--acknowledgments)

## Features

*   **Perspective Warping:** Automatically corrects aerial vehicle tilt (Pitch/Roll) and heading (Yaw) using gimbal telemetry to align oblique aerial footage with top-down satellite maps.
*   **Deep Matching Suite:** Full integration of state-of-the-art matchers.
*   **INS/Odometry Simulation:** Leverages displacement prediction to constrain the search space based on the vehicle's movement, dramatically increasing matching speed and robustness.

## Installation

```bash
# Clone with submodules
git clone --recursive https://github.com/hamitbugrabayram/AerialPositioning.git
cd AerialPositioning

# Setup environment
conda create -n aerialpos python=3.9 -y && conda activate aerialpos
pip install -r requirements.txt

# Download Matcher Weights (Example for MINIMA)
cd matchers/MINIMA/weights && bash download.sh && cd ../../..
```

## Dataset Setup

The pipeline is tested with the **UAV-VisLoc** dataset but supports any georeferenced aerial imagery. For another data check the UAV-VisLoc dataset format!

1.  **Extract Data**: Download UAV-VisLoc dataset.
    ```bash
    unzip UAV_VisLoc_dataset.zip -d _VisLoc_dataset
    ```
2.  **Initialize Central Storage**:
    ```bash
    # Prepare all regions for evaluation across multiple providers and zooms
    python runner.py --dataset-prepare all --zoom-levels 16 17 --tile-provider esri google
    ```
3.  **Unified Structure**:
    ```
    datasets/
    ├── 01_Changjiang_20/
    │   ├── query/                # Global drone images and unified metadata
    │   └── map/
    │       ├── esri/
    │       │   └── 16/           # Cached satellite tiles
    │       └── google/
    ├── ...
    ```

## Usage

The `runner.py` script manages the end-to-end evaluation workflow.

### 1. Dataset Preparation
Fetches satellite tiles and organizes the central data storage.
```bash
python runner.py --dataset-prepare all --zoom-levels 16 17 --tile-provider esri
```

### 2. Visual Positioning
Executes the main positioning pipeline using displacement vector prediction.
```bash
# Run visual positioning for Region 11
python runner.py --dataset-eval 11 --zoom-levels 16 --tile-provider esri --sample-interval 30
```

### 3. Automated Reporting
Generates a consolidated Markdown report with GIFs from the `results/` folder.
```bash
python runner.py --eval-summary all
```
Open `results_report.md` to see the performance metrics and visual tracks.

## Configuration

All system parameters are centralized in `config.yaml`. You can configure the following settings:
*   **Matcher Selection:** Switch between `gim`, `lightglue`, `loftr`, etc.
*   **Model Weights:** Specify paths to different weights of the selected model.
*   **Positioning Parameters:** Adjust `reproj_threshold`, `confidence`, `min_inliers_for_success`, and `max_iter` for geometric verification under `positioning_params`.

## Benchmark Results

The following table summarizes the performance of the visual positioning system on Region 11 (Shandan) using the GIM (LightGlue) matcher across different providers and zoom levels.

| Region (ID) | Provider | Zoom | Success Rate | Avg Error | Avg Inliers | Avg Match Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Shandan (11)** | ESRI | 15 | 65.0% | 30.18m | 479.8 | 0.219s |
| | Google | 15 | 55.0% | 26.25m | 428.5 | 0.040s |
| | Bing | 15 | 60.0% | 30.74m | 413.1 | 0.047s |
| | **ESRI** | **16** | **95.0%** | **27.45m** | **498.0** | **0.073s** |
| | Google | 16 | 90.0% | 25.64m | 518.9 | 0.065s |
| | Bing | 16 | 90.0% | 26.49m | 471.7 | 0.067s |

### Key Insights
*   **Zoom Sensitivity:** Matching success increases dramatically from Zoom 15 (~4.77m/px) to Zoom 16 (~2.38m/px). The higher resolution allows the deep matcher to extract more robust features, leading to a near-perfect success rate (95%).
*   **Provider Consistency:** The system shows remarkably consistent performance across different map providers (ESRI, Google, Bing). This validates the robustness of the Modular Tile Retrieval Engine and the cross-view matching logic.
*   **Search Window Efficiency:** By simulating an INS with Displacement Prediction, the search window is effectively constrained. Even if a frame fails to match, the system predicts the next window correctly, allowing it to find the next best match.
*   **Real-time Potential:** With average match times below 0.1 seconds (at Zoom 16), the pipeline is capable of processing frames at a high frequency, suitable for real-time flight aiding.

## Methodology

### Offline: Modular Tile Retrieval Engine
The system features a decoupled retrieval engine that supports multiple map providers. It leverages the Bing Maps Tile System (QuadKey) and the OpenStreetMap-compatible XYZ system to fetch high-resolution satellite imagery dynamically.

### 1. Perspective Warping (Nadir Transformation)
Oblique aerial vehicle images are transformed into a top-down view using camera intrinsics ($K$) and vehicle rotation ($R$):
$$H_{warp} = K \cdot R_{AV}^T \cdot R_{nadir} \cdot K^{-1}$$
The Adaptive Yaw feature selects the optimal rotation to minimize interpolation artifacts.

### 2. Displacement Prediction (INS Simulation)
In visual positioning, the system predicts the next search window using the aerial vehicle's displacement vector. This simulates an integrated INS/Odometry system, allowing the visual fixes to act as a periodic error resetter that zeros out cumulative drift, reducing the search space to a 1000m local window.

### 3. Dense Matching & RANSAC
Feature correspondences are established using deep transformers. A Homography matrix is estimated via RANSAC with strict stability checks (Determinant validation and boundary constraints).

## Future Enhancements
*   **Full EKF Integration:** Real-time fusion of visual fixes with IMU/Baro data.
*   **Adaptive Zoom Selection:** Automatic zoom level adjustment based on Altitude/GSD.
*   **Recursive Search:** Dynamically expanding search radius on consecutive matching failures.

## References & Acknowledgments
1.  **[UAV-VisLoc Dataset](https://github.com/IntelliSensing/UAV-VisLoc):** Dataset source.
2.  **[WildNav](https://github.com/TIERS/wildnav):** Visual navigation concept.
3.  **[Visual Positioning](https://github.com/TerboucheHacene/visual_localization):** Vision-based GNSS-Free positioning for UAVs in the wild concept.
4.  **Deep Matchers:** [GIM](https://github.com/xuelunshen/gim), [LightGlue](https://github.com/cvg/LightGlue), [LoFTR](https://github.com/zju3dv/LoFTR), [MINIMA](https://github.com/LSXI7/MINIMA).
5.  **Satellite Imagery Providers:** ESRI World Imagery, Google Maps, Bing Maps.

---
*Developed as a B.Sc. Graduation Project.*
