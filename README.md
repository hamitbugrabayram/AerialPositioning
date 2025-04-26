# Visual Localization from Satellites

A visual localization pipeline designed to estimate aerial vehicle coordinates by matching onboard imagery against satellite map tiles at varying zoom levels. The system leverages deep learning-based feature matching and homography estimation to determine precise geolocation.

## Installation

```bash
# Clone with submodules
git clone --recursive https://github.com/ALFONSOBUGRA/SatelliteLocalization.git
cd SatelliteLocalization

# Setup environment
conda create -n satellite-loc python=3.9 -y && conda activate satellite-loc
pip install -r requirements.txt

# Download weights (Required for MINIMA)
cd matchers/MINIMA/weights && bash download.sh && cd -
```

## Methodology

The pipeline estimates the geolocation of an aerial query image $I_q$ relative to a georeferenced satellite map tile $I_m$ through the following stages:

### 1. Preprocessing & Perspective Warping
To handle the viewpoint difference between the oblique aerial view and the nadir satellite view, the system applies a perspective transform. Using the aerial vehicle's telemetry (Yaw, Pitch, Roll) and camera intrinsics $K$, the query image is warped to simulate a top-down view aligned with the satellite map.

### 2. Feature Matching
Deep learning-based matchers (MINIMA, LightGlue, LoFTR) extract keypoints and descriptors to establish correspondences between $I_q$ and $I_m$. This step is robust to scale changes across different satellite zoom levels.

### 3. Geometric Verification
We estimate the Homography matrix $H \in \mathbb{R}^{3 \times 3}$ that maps points from the query image to the map image:

$$
p_{map} \sim H \cdot p_{query}
$$

This is solved using **RANSAC** (Random Sample Consensus) to robustly reject outlier matches based on the reprojection error.

### 4. Geolocation
The center of the query image $c_q$ is projected onto the map coordinate system:

$$
c_{map} = H \cdot c_q
$$

The resulting pixel coordinates $(u, v)$ in $I_m$ are interpolated into GPS coordinates (Latitude, Longitude) using the geodetic bounds of the satellite tile (Haversine projection).

## Data Preparation

**Directory Structure**
```
data/
├── query/
│   ├── photo_metadata.csv  # Aerial telemetry
│   └── *.jpg
└── map/
    ├── map.csv             # Satellite georeference
    └── *.png
```

**Metadata Formats**

*Query Metadata (`photo_metadata.csv`)*
| Column | Description |
|--------|-------------|
| `Filename` | Image filename (e.g., `drone_01.jpg`) |
| `Latitude`, `Longitude` | Ground Truth GPS (for evaluation) |
| `Gimball_Yaw`, `_Pitch`, `_Roll` | Camera angles (degrees) |
| `Flight_Yaw` | Vehicle heading (degrees) |

*Map Metadata (`map.csv`)*
| Column | Description |
|--------|-------------|
| `Filename` | Tile filename (e.g., `tile_01.png`) |
| `Top_left_lat`, `_lon` | Top-left corner coordinates |
| `Bottom_right_lat`, `_long` | Bottom-right corner coordinates |

## Configuration (`config.yaml`)

| Parameter | Description |
|-----------|-------------|
| `matcher_type` | Algorithm: `minima`, `lightglue`, `loftr`, `gim`. |
| `device` | `cuda` (recommended) or `cpu`. |
| `preprocessing.steps` | `['resize', 'warp']`. Order matters. |
| `camera_model` | Intrinsics (Focal length, HFOV) for warping. |
| `ransac_params` | Geometric verification settings (Threshold, Confidence). |

## Usage

**Run Pipeline:**
```bash
python localize.py --config config.yaml
```

**Results:**
Outputs are saved to `data/output/`:
- `localization_results.csv`: Estimated GPS coordinates.
- `processed_queries/`: Preprocessed images.
- `query_X/`: Match visualizations showing inliers.

## References

1.  **LightGlue:** Lindenberger et al. "LightGlue: Local Feature Matching at Light Speed". ICCV 2023. [Code](https://github.com/cvg/LightGlue)
2.  **LoFTR:** Sun et al. "LoFTR: Detector-Free Local Feature Matching with Transformers". CVPR 2021. [Code](https://github.com/zju3dv/LoFTR)
3.  **SuperGlue:** Sarlin et al. "SuperGlue: Learning Feature Matching with Graph Neural Networks". CVPR 2020. [Code](https://github.com/magicleap/SuperGluePretrainedNetwork)
4.  **GIM:** Generalized Image Matching. [Code](https://github.com/xuelunshen/gim)
5.  **MINIMA:** Multi-modal Image Matching. [Code](https://github.com/LSXI7/MINIMA)
