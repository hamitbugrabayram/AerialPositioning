# Satellite Localization: Visual Localization Using Pre-existing Satellite Images 
*Capstone Project | B.Sc. in Astronautical Engineering*

A robust visual localization pipeline for estimating Aerial Vehicle coordinates by matching onboard imagery against satellite map tiles. The system utilizes Aerial Vehicle attitude angles to transform oblique footage into a precise nadir (top-down) view, enabling accurate cross-view matching. 

<p align="center">
  <img src="localization_showcase.gif" width="800" alt="Localization Showcase">
</p>

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage (3-Step Pipeline)](#usage-3-step-pipeline)
  - [Step 1: Dataset Preparation](#step-1-dataset-preparation)
  - [Step 2: Execution](#step-2-execution)
  - [Step 3: Result Evaluation](#step-3-result-evaluation)
- [Methodology](#methodology)
- [Evaluation Results](#evaluation-results)
- [References & Acknowledgments](#references--acknowledgments)

## Features

*   **Perspective Warping:** Automatically corrects aerial vehicle tilt (Pitch/Roll) and heading (Yaw) to align aerial vehicle imagery with satellite imagery.
*   **Different Matchers:** Supports **GIM**, **LightGlue**, **LoFTR**, and **MINIMA**.
*   **Multi-Zoom Support:** Evaluate performance across different satellite resolutions by fetching from Bing. (You can use multiple zoom levels (0-23) for testing)

## Installation

```bash
# Clone with submodules
git clone --recursive https://github.com/hamitbugrabayram/SatelliteLocalization.git
cd SatelliteLocalization

# Setup environment
conda create -n satloc python=3.9 -y && conda activate satloc
pip install -r requirements.txt

# Download Matcher Weights (Example for GIM)
cd matchers/gim && bash download_weights.sh && cd ../..
```

## Dataset Setup

The pipeline is optimized for the **UAV_VisLoc** dataset. 

1.  **Extract Data**: Unzip the provided `UAV_VisLoc_dataset.zip` into the root directory.
    ```bash
    unzip UAV_VisLoc_dataset.zip -d _VisLoc_dataset/
    ```
2.  **Structure**:
    ```
    _VisLoc_dataset/
    ├── 01/
    │   ├── 01.csv            # Contains lat, lon, height, Omega, Phi1, Kappa
    │   └── drone/            # Oblique aerial vehicle images (.jpg)
    ├── 02/ ...
    └── satellite_coordinates_range.csv  # Mapping of IDs to Region Names
    ```

## Usage (3-Step Pipeline)

The system uses `runner.py` to automate the entire workflow.

### Step 1: Dataset Preparation
This step samples **20 random query images (configurable in runner.py)** from each region to ensure a representative but efficient evaluation. It then downloads corresponding satellite tiles from Bing Maps using `src/utils/satellite_retrieval.py` and generates configuration files.
```bash
python runner.py --dataset-prepare
```
*   **Satellite Retrieval:** The system uses a modified version of [Aerial-Satellite-Imagery-Retrieval](https://github.com/Aerial-Satellite-Imagery-Retrieval) to fetch high-resolution tiles based on the sampled query GNSS bounds.

### Step 2: Execution
Runs the localization engine for all prepared regions and zoom levels. For each query image, the system automatically identifies all satellite tiles within a **600-meter radius** to perform the matching process.
```bash
python runner.py --run
```

### Step 3: Result Evaluation
Compiles all results into a single summary report (`experiments_summary.csv`).
```bash
python runner.py --eval
```

## Methodology

### 1. Perspective Warping (Nadir Transformation)
Oblique aerial vehicle images are transformed into a nadir view using Aerial Vehicle attitude angles (Yaw, Pitch, Roll) to eliminate perspective distortion:
$$H_{warp} = K \cdot R_{AV}^T \cdot R_{nadir} \cdot K^{-1}$$

### 2. Spatial Filtering (Relevant Maps)
To optimize matching speed, the system filters the satellite database using a GNSS proximity search. Instead of matching against the entire map collection, it only selects "Relevant Maps":
*   **Haversine Distance:** Calculates the distance between the query's initial GNSS and the center of each satellite tile.
*   **Search Radius:** Only tiles within **600 meters** are passed to the deep-learning matcher.
*   **Optimization:** This reduces the search space from hundreds of tiles to just the immediate neighborhood, significantly speeding up the pipeline.

### 3. Feature Matching
We employ deep-learning matchers (GIM, LightGlue, etc.) to establish dense correspondences between the warped aerial vehicle image and potential satellite tiles. This stage is robust to seasonal changes and scale variations.

### 4. Geometric Verification & Stability
We estimate a Homography matrix $H$ using RANSAC and apply stability filters:
*   **Determinant Check:** $|det(H)| > 10^{-9}$ to avoid singular transformations.
*   **Area Purity:** Ensures the projected footprint area is physically plausible (not extremely skewed or collapsed).
*   **Boundary Constraint:** Predicted center must be within 20% margin of the map tile (`[-0.2, 1.2]` normalized range).

### 5. Precision Geolocation
The pixel-wise center of the aerial vehicle's footprint is projected onto the map tile. We then interpolate this pixel coordinate into GNSS (Lat/Lon) using the **Bing Maps TileSystem (Web Mercator)** projection, which accounts for the Earth's curvature more accurately than linear interpolation.

## Evaluation Results

The following table summarizes the performance of the **GIM** matcher across different regions and zoom levels from the **UAV_VisLoc** dataset. In these experiments, **20 random images** were sampled per region, and matching was restricted to satellite tiles within a **600-meter radius** of the ground truth location. Regions are listed with their average flight altitude to highlight the impact of Ground Sampling Distance (GSD).

| Experiment | Altitude | Zoom | Success Rate | Avg Error | Min Error | Max Error | Median Error | Avg Inliers |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Changjiang_20 (01)** | 406m | 16 | 5.0% | 13.67m | 13.67m | 13.67m | 13.67m | 118.0 |
| | 406m | 17 | 10.0% | 34.52m | 15.40m | 53.64m | 34.52m | 177.0 |
| **Changjiang_23 (02)** | 406m | 16 | 5.0% | 6.13m | 6.13m | 6.13m | 6.13m | 283.0 |
| | 406m | 17 | 15.0% | 13.82m | 6.81m | 24.40m | 10.25m | 272.7 |
| **Taizhou_1 (03)** | 466m | 16 | 20.0% | 15.76m | 3.44m | 27.22m | 16.18m | 193.0 |
| | 466m | 17 | 50.0% | 21.13m | 3.29m | 32.59m | 22.99m | 280.6 |
| **Taizhou_6 (04)** | 545m | 16 | 30.0% | 42.39m | 13.06m | 97.01m | 31.91m | 290.7 |
| | 545m | 17 | 45.0% | 30.10m | 12.56m | 96.48m | 22.68m | 349.0 |
| **Yunnan (05)** | 2315m | 16 | 5.0% | 46.50m | 46.50m | 46.50m | 46.50m | 130.0 |
| | 2315m | 17 | 15.0% | 42.80m | 27.30m | 54.95m | 46.15m | 264.3 |
| **Zhuxi (06)** | 838m | 16 | 0.0% | N/A | N/A | N/A | N/A | 0.0 |
| | 838m | 17 | 0.0% | N/A | N/A | N/A | N/A | 0.0 |
| **Donghuayuan (07)** | 690m | 16 | 0.0% | N/A | N/A | N/A | N/A | 0.0 |
| | 690m | 17 | 5.0% | 2.78m | 2.78m | 2.78m | 2.78m | 123.0 |
| **Huzhou_3_P1 (08)** | 552m | 16 | 35.0% | 27.00m | 13.04m | 39.99m | 23.18m | 311.6 |
| | 552m | 17 | 50.0% | 27.62m | 13.69m | 55.60m | 24.28m | 370.8 |
| **Huzhou_3_P2 (09)** | 547m | 16 | 25.0% | 20.64m | 4.10m | 30.91m | 26.26m | 490.4 |
| | 547m | 17 | 50.0% | 23.81m | 4.49m | 40.27m | 24.53m | 290.0 |
| **Huailai (10)** | 772m | 16 | 0.0% | N/A | N/A | N/A | N/A | 0.0 |
| | 772m | 17 | 20.0% | 23.97m | 20.50m | 26.59m | 24.40m | 158.5 |
| **Shandan (11)** | 2572m | 16 | 90.0% | 29.95m | 11.74m | 82.91m | 24.24m | 412.6 |
| | 2572m | 17 | 65.0% | 27.61m | 11.38m | 52.17m | 26.76m | 247.3 |

### Key Observations
*   **Altitude vs. Zoom:** Higher altitude regions (e.g., Shandan @ 2572m) show very high success rates at Zoom 16 (90%), as the Ground Sampling Distance (GSD) of the aerial vehicle image aligns perfectly with lower satellite resolutions.
*   **Precision:** Successful matches typically yield an average error between **5m and 30m**, demonstrating the effectiveness of the perspective warping and Mercator projection.

### Future Enhancements
The project can be further advanced by implementing the following improvements:
*   **Advanced Map Search:** Implementing more efficient global or local map tile search algorithms to reduce the dependency on initial GNSS proximity.
*   **Dead Reckoning with VIO:** Integrating Visual-Inertial Odometry (VIO) to maintain pose estimation in intervals where visual matching with satellite imagery fails.
*   **Visual Place Recognition (VPR):** Supporting the system with VPR models to enhance robust global localization in feature-poor or repetitive environments.
*   **Sophisticated Filtering:** Implementing more advanced statistical or learning-based filtering methods (e.g., Kalman Filters, Particle Filters) to smooth the trajectory and eliminate localization outliers.


## References & Acknowledgments

1.  **Core Concept Inspiration:** [WildNav (TIERS)](https://github.com/TIERS/wildnav) - Conceptual idea of cross-view visual localization.
2.  **Dataset:** [UAV-VisLoc Dataset](https://github.com/IntelliSensing/UAV-VisLoc) - Official repository for the UAV-VisLoc benchmark.
3.  **Tile Retrieval:** [Aerial-Satellite-Imagery-Retrieval](https://github.com/Aerial-Satellite-Imagery-Retrieval) - Used for fetching Bing Maps satellite tiles.
4.  **Matchers:** GIM, LightGlue, LoFTR, SuperGlue.

---
*Developed as a graduation project.*
