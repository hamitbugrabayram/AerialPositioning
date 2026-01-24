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

The following table summarizes the performance of the **GIM** matcher across different regions and zoom levels. In these experiments, **20 random images** were sampled per region. Regions are listed with their average flight altitude to highlight the impact of Ground Sampling Distance (GSD).

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

### Key Observations
*   **Optimal Zoom Window:** Most regions show peak performance at **Zoom 17**, where the success rate is significantly higher than at lower (Zoom 15) or higher (Zoom 18) levels. This suggests that Zoom 17 provides the best balance of context and detail for mid-altitude (400m-800m) flights.
*   **High Altitude Robustness:** High-altitude regions like **Shandan (2572m)** maintain high success rates (up to 85%) even at lower zoom levels (Zoom 15/16). This is because the wide FOV from high altitudes covers more ground area, matching the larger scale of low-zoom satellite tiles.
*   **Precision Extremes:** In successful matches, we observe a wide error range. **Donghuayuan** achieved exceptional precision (1.75m avg error), while other regions stabilized around **20-30m**, which is still within a reliable range for GNSS-denied navigation.
*   **Failure at Zoom 15:** For lower altitude flights, Zoom 15 often results in 0% success. The pixel resolution at this level is likely too coarse to capture the distinct features needed by deep-learning matchers for accurate alignment.

### Future Enhancements
The project can be further advanced by implementing the following improvements (I plan to implement these as time permits):
*   **Adaptive Zoom Selection:** Automatically determining the optimal satellite zoom level based on the aircraft's real-time altitude and Ground Sampling Distance (GSD).
*   **Map Database & Efficient Retrieval:** Developing a robust map database system to streamline searching and fetching appropriate satellite tiles.
*   **Recursive Search Expansion:** Implementing a dynamic search strategy that expands the search radius incrementally if a match cannot be found in the initial area.
*   **Dead Reckoning (Odometry/VIO):** Integrating Odometry or Visual-Inertial Odometry (VIO) to maintain pose estimation in intervals where visual matching with satellite imagery fails.
*   **Sophisticated Filtering:** Implementing more advanced statistical or learning-based filtering methods (e.g., Kalman Filters, Particle Filters) to smooth the trajectory and eliminate localization outliers.


## References & Acknowledgments

1.  **Core Concept Inspiration:** [WildNav (TIERS)](https://github.com/TIERS/wildnav) - Conceptual idea of cross-view visual localization.
2.  **Dataset:** [UAV-VisLoc Dataset](https://github.com/IntelliSensing/UAV-VisLoc) - Official repository for the UAV-VisLoc dataset.
3.  **Tile Retrieval:** [Aerial-Satellite-Imagery-Retrieval](https://github.com/Aerial-Satellite-Imagery-Retrieval) - Used for fetching Bing Maps satellite tiles.

---
*Developed as a graduation project.*
