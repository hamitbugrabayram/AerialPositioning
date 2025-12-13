<div align="center">

<img src="thumbnail.png" alt="Satellite Visual Localization" width="700"/>

# 🛰️ Satellite Visual Localization

**Drone Position Estimation System using Satellite Imagery**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Features](#-features) •
[Installation](#-installation) •
[Quick Start](#-quick-start) •
[Configuration](#%EF%B8%8F-configuration) •
[Results](#-results) •
[Contributing](#-contributing)

</div>

---

## 📋 Overview

This repository provides a **drone position estimation system** that determines the GPS coordinates of a drone by matching drone camera images against satellite map tiles. Using state-of-the-art feature matching algorithms, the system calculates precise geo-localization with meter-level accuracy.

### 🎯 Key Capabilities

- **Position Estimation**: Estimate drone GPS coordinates from camera imagery
- **Cross-view Matching**: Match drone/UAV images against satellite map tiles
- **Meter-level Accuracy**: Precise localization using Haversine distance computation
- **Multiple Matchers**: Support for various feature matching algorithms (optional)

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🔧 Supported Matchers (Optional)

| Matcher | Type | Description |
|---------|------|-------------|
| **LightGlue** | Sparse | Fast and accurate local feature matching |
| **SuperGlue** | Sparse | Graph neural network-based matcher |
| **LoFTR** | Dense | Detector-free local feature matching |
| **GIM** | Various | Generalized image matching framework |

</td>
<td width="50%">

### 📊 Output Results

- ✅ Estimated GPS coordinates (latitude, longitude)
- ✅ Localization error (meters)
- ✅ Match quality metrics
- ✅ Per-image detailed results
- ✅ Overall statistics

</td>
</tr>
</table>

### 🔄 Preprocessing Pipeline

- **Resizing**: Configurable image dimension constraints
- **Perspective Warping**: Simulate nadir (top-down) view from oblique angles
- **Camera Model**: Support for custom camera intrinsics

---

## 🚀 Installation

### Prerequisites

- **CUDA-capable GPU** (recommended for optimal performance)
- **Conda** package manager ([Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/))

### Step 1: Clone Repository

```bash
# Clone with submodules (required)
git clone --recursive https://github.com/ALFONSOBUGRA/SatelliteLocalization.git
cd SatelliteLocalization
```

<details>
<summary>📌 Already cloned without <code>--recursive</code>?</summary>

```bash
git submodule update --init --recursive
```

</details>

### Step 2: Create Environment

```bash
# Create and activate conda environment
conda create -n satellite-loc python=3.9 -y
conda activate satellite-loc

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Download Model Weights

Download the pretrained weights for your desired matcher(s):

| Matcher | Weights Location | Notes |
|---------|-----------------|-------|
| LightGlue | Auto-downloaded | SuperPoint or DISK features |
| SuperGlue | Auto-downloaded | Indoor/Outdoor variants |
| LoFTR | `matchers/LoFTR/weights/` | [Download outdoor_ds.ckpt](https://github.com/zju3dv/LoFTR) |
| GIM | `matchers/gim/weights/` | [Download from GIM repo](https://github.com/xuelunshen/gim) |

---

## ⚡ Quick Start

### 1. Prepare Your Data

```
data/
├── query/
│   ├── photo_metadata.csv    # Query image metadata
│   └── *.jpg                 # Drone/UAV images
└── map/
    ├── map.csv               # Satellite tile metadata
    └── *.png                 # Satellite map tiles
```

<details>
<summary>📄 <strong>Metadata Format Reference</strong></summary>

**Query Metadata** (`photo_metadata.csv`):
```csv
Filename,Latitude,Longitude,Gimball_Yaw,Gimball_Pitch,Gimball_Roll,Flight_Yaw
image001.jpg,41.0082,28.9784,-5.2,-85.0,0.0,45.0
```

**Map Metadata** (`map.csv`):
```csv
Filename,Top_left_lat,Top_left_lon,Bottom_right_lat,Bottom_right_long
tile_001.png,41.0100,28.9750,41.0050,28.9850
```

</details>

### 2. Configure Settings

Edit `config.yaml` to select your matcher and parameters:

```yaml
matcher_type: 'lightglue'  # Options: lightglue, superglue, loftr, gim
device: 'cuda'

preprocessing:
  enabled: true
  steps: ['resize', 'warp']
  resize_target: [1024]
```

### 3. Run Localization

```bash
python localize.py --config config.yaml
```

---

## ⚙️ Configuration

The `config.yaml` file provides comprehensive control over the localization:

<details>
<summary><strong>🔍 Full Configuration Reference</strong></summary>

```yaml
# Matcher Selection
matcher_type: 'lightglue'    # lightglue | superglue | loftr | gim
device: 'cuda'               # cuda | cpu

# Data Paths
data_paths:
  query_dir: 'data/query'
  map_dir: 'data/map'
  output_dir: 'data/output'
  query_metadata: 'data/query/photo_metadata.csv'
  map_metadata: 'data/map/map.csv'

# Preprocessing
preprocessing:
  enabled: true
  steps: ['resize', 'warp']
  resize_target: [1024]
  target_gimbal_pitch: -90.0  # Nadir view

# Camera Model (for perspective warping)
camera_model:
  focal_length: 4.5
  resolution_width: 3040
  resolution_height: 4056
  hfov_deg: 82.9

# RANSAC Parameters
ransac_params:
  method: 'RANSAC'
  reproj_threshold: 8.0
  confidence: 0.999
  max_iter: 10000

# Localization Settings
localization_params:
  save_visualization: true
  min_inliers_for_success: 10
```

</details>

---

## 📈 Results

Results are saved to a timestamped directory in `data/output/`:

```
data/output/lightglue_preprocessed_20240115-143052/
├── localization_results.csv   # Per-query localization results
├── localization_stats.txt     # Overall statistics
├── processed_queries/         # Preprocessed query images
└── query_001/
    ├── query_001_vs_tile_001_results.txt
    └── query_001_vs_tile_001_match.png
```

### Sample Output

| Metric | Description |
|--------|-------------|
| `Pred Latitude` | Estimated latitude coordinate |
| `Pred Longitude` | Estimated longitude coordinate |
| `Error (m)` | Distance between ground truth and prediction |
| `Success` | Whether localization was successful |

---

## 🏗️ Project Structure

```
SatelliteLocalization/
├── 📄 localize.py            # Main localization script
├── 📄 config.yaml            # Configuration file
├── 📄 requirements.txt       # Python dependencies
├── 📁 src/                   # Source code
│   ├── lightgluePipeline.py  # LightGlue matcher
│   ├── supergluePipeline.py  # SuperGlue matcher
│   ├── loftrPipeline.py      # LoFTR matcher
│   ├── gimPipeline.py        # GIM matcher
│   └── utils/                # Utility modules
│       ├── helpers.py        # GPS calculations
│       ├── preprocessing.py  # Image preprocessing
│       └── visualization.py  # Match visualization
├── 📁 matchers/              # Matcher submodules
│   ├── LightGlue/
│   ├── SuperGluePretrainedNetwork/
│   ├── LoFTR/
│   └── gim/
└── 📁 data/                  # Data directory
    ├── query/                # Query images
    └── map/                  # Satellite tiles
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📚 Citation

If you use this system in your research, please cite:

```bibtex
@software{satellite_localization,
  title = {Satellite Visual Localization},
  author = {ALFONSOBUGRA},
  url = {https://github.com/ALFONSOBUGRA/SatelliteLocalization},
  year = {2024}
}
```

---

## 🙏 Acknowledgments

This system builds upon excellent open-source work:

- [LightGlue](https://github.com/cvg/LightGlue) - ETH Zurich
- [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) - Magic Leap
- [LoFTR](https://github.com/zju3dv/LoFTR) - Zhejiang University
- [GIM](https://github.com/xuelunshen/gim) - Generalized Image Matching
- [WildNav](https://github.com/research/wildnav) - Conceptual inspiration

---

<div align="center">

**Made with ❤️ for the computer vision community**

</div>
