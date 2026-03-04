# Aerial Positioning: Visual Positioning for Aerial Imagery Using Pre-existing Satellite Images
*Developed as a Capstone Project | B.Sc. in Astronautical Engineering*

This repository presents a vision-based positioning process for estimating the horizontal position (latitude and longitude) of an aerial platform in GNSS-denied environments. Given only an initial starting position, the method matches onboard imagery against pre-existing satellite map tiles to produce coordinate estimates purely from visual information. To improve cross-view consistency, the process uses aerial vehicle attitude information to rectify oblique camera views into a nadir-oriented perspective aligned with satellite imagery.

<p align="center">
  <img src="assets/zoom16_combined_showcase.gif">
  <br>
  <i>Ground Truth (Green) vs. Estimated Position (Blue)</i>
</p>

## Table of Contents
- [Environment Setup and Usage](#environment-setup-and-usage)
- [Methodology](#methodology)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Limitations and Future Work](#limitations-and-future-work)
- [References & Acknowledgments](#references--acknowledgments)

## Environment Setup and Usage

### 1. Environment Setup

Use Python 3.9 with dependencies from `requirements.txt`. For complete matcher support, clone with `--recursive` and download the required model weights.

### 2. Dataset and Directory Layout

The default setup uses the **[UAV-VisLoc dataset](https://github.com/IntelliSensing/UAV-VisLoc)**. After downloading the archive, extract raw data under `_VisLoc_dataset/` by running:

```bash
unzip UAV_VisLoc_dataset.zip -d _VisLoc_dataset
```

Then run the preparation step to build unified region folders under `datasets/`.

Expected layout after preparation:

```text
datasets/
├── 01_Changjiang_20/
│   ├── query/                
│   └── map/
│       ├── esri/
│       │   └── 16/
│       └── google/
├── ...
```

### 3. Standard Evaluation Commands

```bash
# Prepare all regions for multiple providers and zoom levels
python runner.py --dataset-prepare all --zoom-levels 16 17 --tile-provider esri google

# Evaluate selected region(s)
python runner.py --dataset-eval 11 --zoom-levels 16 --tile-provider esri

# Aggregate reports
python runner.py --eval-summary all
```

## Methodology

### Pipeline Overview
*   **Offline Tile Retrieval:** A multi-provider retrieval module pre-downloads and caches georeferenced tiles (ESRI/Google) before runtime.
*   **Query Preprocessing:** Drone images are resized and perspective-warped to a nadir, north-oriented view using only onboard IMU/gimbal sensor readings (roll, pitch, yaw). No GPS coordinates are used.
*   **Adaptive Tile Stitching:** Neighbouring satellite tiles are composed into an NxN grid based on barometric altitude and zoom level so the reference image covers the drone's estimated ground footprint.
*   **Intra-frame Adaptive Exponential Backoff Search:** Given an initial position, the search center tracks the last successful match. If a match fails, the search radius dynamically grows exponentially (e.g., x2.0) *within the same frame* up to a max limit to quickly recover the platform. Upon success, the excess radius cools down toward the initial value for subsequent frames.
*   **Deep Matching and Geometric Validation:** Transformer-based correspondences (MINIMA SuperPoint+LightGlue) are validated through RANSAC and geometric plausibility constraints.

### 1. Offline Tile Retrieval
Satellite maps are downloaded offline prior to execution. A decoupled tile retrieval engine supports multiple providers (ESRI, Google), enabling high-resolution tiles for the region of interest to be cached in advance and served without live network access during positioning.

### 2. Query Preprocessing (Perspective Rectification)
Oblique UAV imagery is rectified to a nadir (top-down), north-facing view to establish geometric comparability with ortho-rectified satellite tiles.

The preprocessing pipeline uses **only onboard sensor data** (no GPS/GT coordinates):

1. **Resize** to a standard dimension (max 768px).
2. **Perspective warp** using gimbal angles from the IMU:
   - **Yaw** (Phi1): heading direction from the gimbal encoder.
   - **Pitch**: elevation angle from the gimbal encoder.
   - **Roll**: bank angle from the gimbal encoder.

The intrinsic matrix $K$ is estimated from image dimensions ($f = \max(w, h)$, principal point at center) since real camera intrinsics are not available in the dataset. The warp homography is:

$$H_{warp} = K \cdot R_{current}^T \cdot R_{nadir} \cdot K^{-1}$$

where $R_{current}$ is built from the gimbal Euler angles and $R_{nadir}$ targets a straight-down, north-aligned view ($\text{pitch} = -90°$, $\text{yaw} = 0°$, $\text{roll} = 0°$).

### 3. Adaptive Tile Stitching
A single 256x256 satellite tile may not cover the drone's full field of view, especially at high altitudes. The engine computes an adaptive grid size from:

- **Barometric altitude** (available without GPS) and a coverage factor.
- **Map tile latitude** (from the tile's own georeferencing, not from query GPS).
- **Zoom level** (determines tile ground resolution).

$$\text{grid} = \min\!\left(\left\lceil \frac{\text{altitude} \times \text{coverage\_factor}}{\text{tile\_ground\_coverage}}\right\rceil,\; \text{max\_grid}\right)$$

With `max_grid=3`, the composite is 768x768 px (3x3 tiles), which matches the matcher's internal resize target and preserves detail. Missing edge tiles are filled with neutral gray.

### 4. Adaptive Exponential Backoff Search
The search center is initialized from the platform's known starting position. On each frame, candidate satellite tiles are filtered around the current search center using an **intra-frame adaptive exponential backoff** radius:

* **On failure:** the search radius dynamically expands by a growth factor ($r \leftarrow \min(r \cdot f_{grow},\; r_{max})$) and matching is re-attempted *on the same query image* until it succeeds or hits the cap. This allows the system to recover from long GNSS-denied gaps or visual mismatches.
* **On success:** the search center snaps to the new matched position, and the excess radius above the initial value decays by a cooldown factor ($r \leftarrow r_0 + (r - r_0) \cdot f_{cool}$) for the next frame.

Default parameters: $r_0 = 1000\text{ m}$, $r_{max} = 4000\text{ m}$, $f_{grow} = 2.0$, $f_{cool} = 0.5$.

### 5. Deep Matching and Geometric Verification
Dense or semi-dense correspondences are computed using the MINIMA framework (SuperPoint + LightGlue). A planar homography $H$ between query and reference tile is then estimated via RANSAC. Acceptance is conditioned on stability checks, including a determinant constraint ($|\det H| \approx 1$) for near-rigid behavior, image-boundary consistency of projected corners, and non-degeneracy of the estimated transformation.

## Evaluation Setup

The results reported below correspond to the following setup:

*   **Matcher:** MINIMA (SuperPoint + LightGlue).
*   **Tile Provider:** Google.
*   **Preprocessing:** Resize + gimbal warp (Phi1 as yaw, estimated K from image dims).
*   **Map Context:** Adaptive NxN grid stitching (`coverage_factor=2.0`, `max_grid=3`).
*   **Temporal Sampling:** `sample_interval=1` (all frames evaluated).
*   **Search Strategy:** Intra-frame adaptive backoff (`initial=1000m`, `max=4000m`, `growth=2.0`, `cooldown=0.5`).
*   **Geometric Acceptance:** `min_inliers_for_success=50`.

### Sensor Data Used at Runtime

| Data | Source | GPS Required? |
| :--- | :--- | :--- |
| Gimbal Roll / Pitch / Yaw | IMU / gimbal encoder | No |
| Barometric altitude | Barometer / IMU fusion | No |
| Initial search center | First-frame coordinate | One-time only |

No ground-truth GPS coordinates are used during matching or positioning. Altitude and gimbal angles are onboard sensor measurements available in any GNSS-denied scenario.

## Results

### Sanity Validation (first 20 frames per region)

| Region (ID) | Zoom | Success Rate | Avg Inliers | Median Error |
| :--- | :--- | :--- | :--- | :--- |
| Changjiang_20 (01) | 18 | **100%** (20/20) | 367 | 23.2 m |
| Changjiang_23 (02) | 18 | **100%** (20/20) | 453 | 14.4 m |
| Taizhou_1 (03) | 18 | **100%** (20/20) | 395 | 13.1 m |
| Yunnan (05) | 17 | **100%** (20/20) | 215 | 19.9 m |
| Shandan (11) | 16 | **100%** (20/20) | 287 | 18.3 m |

### Previous Full Evaluation (Shandan, GIM matcher)

| Region (ID) | Provider | Zoom | Success Rate | Avg Error | Avg Inliers |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Shandan (11) | ESRI | 15 | 70.51% | 29.84m | 354.6 |
| Shandan (11) | Google | 15 | 74.75% | 30.96m | 342.2 |
| Shandan (11) | ESRI | 16 | 98.14% | 29.93m | 521.5 |
| Shandan (11) | Google | 16 | **99.32%** | 30.06m | 538.6 |

### Zoom 16 Error Evolution
<p align="center">
  <img src="assets/zoom16_combined_error_plot.png" width="1000" alt="Zoom 16 Error Evolution">
</p>

## Limitations and Future Work

### Current Limitations
*   **Initial Position Requirement:** The system requires a known starting coordinate to initialize the search center.
*   **Planar Scene Assumption:** Homography-based localization is most reliable when local scene geometry is approximately planar.
*   **Map Dependency:** Performance depends on satellite-tile quality, seasonal consistency, and provider-specific appearance.

### Future Work
*   **Full EKF Integration:** Tightly/loosely coupled fusion of visual position fixes with IMU and barometric measurements in a unified state estimator.
*   **DTED-Assisted 3D Geolocation (PnP):** Incorporate DTED/elevation priors and solve a Perspective-n-Point (PnP) problem to estimate full geodetic state (`latitude`, `longitude`, `altitude`) instead of horizontal position only.
*   **Model Optimization and Edge Deployment:** Quantization/pruning and export to ONNX/TensorRT for embedded real-time deployment.

## References & Acknowledgments
1. **[UAV-VisLoc Dataset](https://github.com/IntelliSensing/UAV-VisLoc):** Dataset source.
2. **[WildNav](https://github.com/TIERS/wildnav):** Visual navigation concept.
3. **[Visual Localization](https://github.com/TerboucheHacene/visual_localization):** Vision-based GNSS-free localization concept.
4. **Deep Matchers:** [GIM](https://github.com/xuelunshen/gim), [LightGlue](https://github.com/cvg/LightGlue), [LoFTR](https://github.com/zju3dv/LoFTR), [MINIMA](https://github.com/LSXI7/MINIMA).
5. **Satellite Imagery Providers:** ESRI World Imagery, Google Maps.
